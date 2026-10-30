# -*- coding: utf-8 -*-
"""Pruebas sin entrenar con dos modelos ya entrenados, uno de color y otro con fusión (D-04). Generaliza
`combinar_e1_e2.py` (D-03), que se conserva tal cual, con un tope para α:

- interruptor: el de color si la imagen tiene menos de `umbral` de profundidad válida; el de fusión si no.
- mezcla: p = (1 − α)·p_color + α·p_fusión por píxel, con α = a · (fracción válida en una ventana de k×k celdas
  de 1/8). Con a = 1 es la mezcla de D-03; con a < 1, aun con profundidad completa los dos modelos se promedian,
  como hace la compuerta de E8 (α medio 0.59 con profundidad completa).
- relleno: cada hueco toma la profundidad del píxel válido más cercano y p = (1 − a)·p_color + a·p_fusión.

`umbral`, k y a se eligen en la validación por escenas, con los modelos de validación y la media de todas las
condiciones; el test se evalúa una vez, con las 3 semillas (color y fusión de la misma semilla).

Control del ensamble: el promedio de dos modelos de color de semillas distintas (pares 0-1, 1-2 y 0-2), sin
profundidad, para separar lo que aporta la profundidad de lo que aporta promediar dos modelos.

    python src/combinar.py --color E1u --fusion E2u

Resultado en runs/rgbd/combinaciones/<color>+<fusion>/. Hasta D-05 se escribía `resultado.json` y, si cambiaba la
lr, el anterior se renombraba; desde D-06 los resultados existentes no se tocan: cada combinación nueva va a
`resultado_<interpolación>_<color>_lr<lr>+<fusión>_lr<lr>.json`, y si ese archivo ya existe no se repite.

Interpolación (D-06). Hasta D-05 se promediaban las probabilidades a 1/8 y después se ampliaban a 480×640
(`--interpolar probabilidades`): así un modelo solo da algo distinto que en `entrenar_rgbd.evaluar`, que amplía
los logits (E1u: 47.27 frente a 47.31). Por defecto ahora se amplían los logits de cada modelo, se pasa a
probabilidades y se mezcla a 480×640 (`--interpolar logits`); α se amplía bilinealmente. Con un solo modelo
coincide exactamente con `evaluar`, así que las cifras de los modelos solos y las de las mezclas son comparables.

Control del ensamble: los tres pares de semillas (0-1, 1-2, 0-2) comparten modelos, así que no son tres réplicas
independientes; se informan los tres valores y su rango, no un intervalo.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
from combinar_e1_e2 import UMBRALES, VENTANAS, rellenar  # noqa: E402
from entrenar_rgbd import (DEV, IGNORAR, RUNS, autocast, cargar, elegida, lote_a_gpu, nombre_corrida,  # noqa: E402
                           normalizar_rgb)

TOPES = (0.5, 0.6, 0.7, 0.8, 1.0)
PARES = ((0, 1), (1, 2), (0, 2))


def fraccion(valida, tam, k):
    m = F.adaptive_avg_pool2d(valida.float(), tam)
    return F.avg_pool2d(m, k, 1, k // 2, count_include_pad=False) if k > 1 else m


def probabilidades(lg, tam, interpolar):
    """Logits a 1/8 -> probabilidades: a 1/8 («probabilidades», hasta D-05) o a `tam` («logits», D-06)."""
    if interpolar == "logits":
        lg = F.interpolate(lg.float(), tam, mode="bilinear", align_corners=False)
    return lg.float().softmax(1)


def al_tamano(m, p):
    """α (B,1,h,w) a la resolución de p (sin cambios si ya coincide)."""
    return m if m.shape[-2:] == p.shape[-2:] else F.interpolate(m, p.shape[-2:], mode="bilinear",
                                                                 align_corners=False)


def mezclar(p_color, p_fusion, alfa):
    """p = (1 − α)·p_color + α·p_fusión. Con α = 0 devuelve p_color exactamente (respaldo a la ruta de color)."""
    alfa = al_tamano(alfa, p_color)
    return torch.where(alfa > 0, (1 - alfa) * p_color + alfa * p_fusion, p_color)


def recorrer(m1, m2, datos, posiciones, lote=4, interpolar="probabilidades"):
    """mIoU de cada variante en cada condición: {condición: {variante: mIoU}}."""
    m1.eval(); m2.eval()
    conf = {}

    def sumar(cond, var, p, et):
        pred = F.interpolate(p, et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        m = et != IGNORAR
        c = conf.setdefault((cond, var), torch.zeros(40, 40, dtype=torch.long, device=DEV))
        c += torch.bincount(40 * et[m] + pred[m], minlength=1600).view(40, 40)

    for cond in rgbd.CONDICIONES:
        for k in range(0, len(posiciones), lote):
            idx = np.sort(posiciones[k:k + lote])
            rgb, _, et = lote_a_gpu(datos, idx)
            pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in idx]
            prof = torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None]
            valida = torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None]
            prof_r = torch.from_numpy(np.stack([rellenar(p, v) for p, v in pv])).to(DEV)[:, None]
            x = normalizar_rgb(rgb)
            tam = et.shape[-2:]
            with torch.no_grad(), autocast():
                l1, l2, l2r = m1(x, prof, valida), m2(x, prof, valida), m2(x, prof_r, prof_r > 0)
            p1, p2, p2r = (probabilidades(l, tam, interpolar) for l in (l1, l2, l2r))
            sumar(cond, "color", p1, et)
            sumar(cond, "fusion", p2, et)
            frac = valida.float().mean((1, 2, 3)).view(-1, 1, 1, 1)
            for u in UMBRALES:
                sumar(cond, f"interruptor_{u:g}", torch.where(frac >= u, p2, p1), et)
            for v in VENTANAS:
                m = fraccion(valida, l1.shape[-2:], v)       # α se calcula a 1/8, como la compuerta de E8
                for a in TOPES:
                    if interpolar == "logits":
                        sumar(cond, f"mezcla_{v}_{a:g}", mezclar(p1, p2, a * m), et)
                    else:                                    # la fórmula de D-04, tal cual
                        sumar(cond, f"mezcla_{v}_{a:g}", (1 - a * m) * p1 + a * m * p2, et)
            for a in TOPES:
                sumar(cond, f"relleno_{a:g}", (1 - a) * p1 + a * p2r, et)
    res = {}
    for (cond, var), c in conf.items():
        c = c.double()
        iou = c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)
        res.setdefault(cond, {})[var] = float(100 * iou.mean())
    return res


def ensamble_color(color, lr, datos, lote=4, interpolar="probabilidades"):
    """mIoU del promedio de dos modelos de color de semillas distintas, para cada par de `PARES`."""
    out = []
    for a, b in PARES:
        ma, _ = cargar(RUNS / nombre_corrida(color, lr, a))
        mb, _ = cargar(RUNS / nombre_corrida(color, lr, b))
        ma.eval(); mb.eval()
        conf = torch.zeros(40, 40, dtype=torch.long, device=DEV)
        for k in range(0, len(datos["rgb"]), lote):
            rgb, _, et = lote_a_gpu(datos, np.arange(k, min(k + lote, len(datos["rgb"]))))
            x = normalizar_rgb(rgb)
            with torch.no_grad(), autocast():
                la, lb = ma(x), mb(x)
            p = (probabilidades(la, et.shape[-2:], interpolar) + probabilidades(lb, et.shape[-2:], interpolar)) / 2
            pred = F.interpolate(p, et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            m = et != IGNORAR
            conf += torch.bincount(40 * et[m] + pred[m], minlength=1600).view(40, 40)
        c = conf.double()
        out.append(float(100 * (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)).mean()))
        del ma, mb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="E1")
    ap.add_argument("--fusion", default="E2")
    ap.add_argument("--interpolar", choices=("logits", "probabilidades"), default="logits")
    ap.add_argument("--lote", type=int, default=2)
    a = ap.parse_args()
    (lr1, _), _ = elegida(a.color)
    (lr2, _), _ = elegida(a.fusion)

    carpeta = RUNS / "combinaciones" / f"{a.color}+{a.fusion}"
    archivo = carpeta / f"resultado_{a.interpolar}_{a.color}_lr{lr1:g}+{a.fusion}_lr{lr2:g}.json"
    if archivo.exists():        # D-06: nunca se sobrescribe ni se renombra un resultado
        print(f"{archivo} ya existe; no se repite", flush=True)
        return
    kw = {"lote": a.lote, "interpolar": a.interpolar}
    tr = nyuv2.cache("entrenamiento")
    _, pos_val = nyuv2.validacion_por_escenas()
    m1, _ = cargar(RUNS / nombre_corrida(a.color, lr1, 0, validacion=True))
    m2, _ = cargar(RUNS / nombre_corrida(a.fusion, lr2, 0, validacion=True))
    val = recorrer(m1, m2, tr, pos_val, **kw)
    del m1, m2
    media = lambda var: float(np.mean([val[c][var] for c in rgbd.CONDICIONES]))  # noqa: E731
    umbral = max(UMBRALES, key=lambda u: media(f"interruptor_{u:g}"))
    mezcla = max((f"mezcla_{v}_{t:g}" for v in VENTANAS for t in TOPES), key=media)
    relleno = max((f"relleno_{t:g}" for t in TOPES), key=media)
    print(f"{a.color}+{a.fusion}, validación (media de las condiciones): color {media('color'):.2f}, "
          f"fusión {media('fusion'):.2f}, interruptor_{umbral:g} {media(f'interruptor_{umbral:g}'):.2f}, "
          f"{mezcla} {media(mezcla):.2f}, {relleno} {media(relleno):.2f}", flush=True)

    test = nyuv2.cache("test")
    variantes = ["color", "fusion", f"interruptor_{umbral:g}", mezcla, relleno]
    por_semilla = []
    for s in range(3):
        m1, _ = cargar(RUNS / nombre_corrida(a.color, lr1, s))
        m2, _ = cargar(RUNS / nombre_corrida(a.fusion, lr2, s))
        por_semilla.append(recorrer(m1, m2, test, np.arange(len(test["rgb"])), **kw))
        del m1, m2
        print(f"semilla {s} evaluada", flush=True)
    tabla = {c: {v: [r[c][v] for r in por_semilla] for v in variantes} for c in rgbd.CONDICIONES}
    control = ensamble_color(a.color, lr1, test, **kw)
    out = {"color": [a.color, lr1], "fusion": [a.fusion, lr2], "validacion": val, "umbral": umbral,
           "mezcla": mezcla, "relleno": relleno, "variantes": variantes,
           "test": {c: {v: {"media": st.mean(x), "de": st.stdev(x), "semillas": x} for v, x in fila.items()}
                    for c, fila in tabla.items()},
           "ensamble_color": {"pares": PARES, "miou": control, "media": st.mean(control), "de": st.stdev(control),
                              "nota": "pares solapados: no son réplicas independientes"},
           "interpolar": a.interpolar, "evaluacion": "exploratoria si el test ya se usó para decidir (D-06)"}
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'condición':14s}" + "".join(f"{v:>16s}" for v in variantes))
    for c, fila in tabla.items():
        print(f"{c:14s}" + "".join(f"{st.mean(x):10.2f} ±{st.stdev(x):4.2f}" for x in fila.values()))
    print(f"ensamble de dos {a.color} (sin profundidad): {st.mean(control):.2f} ± {st.stdev(control):.2f}")


if __name__ == "__main__":
    main()
