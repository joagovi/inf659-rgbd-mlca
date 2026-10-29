# -*- coding: utf-8 -*-
"""Pruebas sin entrenar con los modelos ya entrenados de E1 (solo color) y E2 (suma simple), D-03:

- interruptor: E1 si la imagen tiene menos de `umbral` de profundidad válida; E2 si no.
- mezcla: p = (1 − α)·p_E1 + α·p_E2 por píxel, con α = fracción válida en una ventana de k×k celdas de 1/8.
  Es E8 sin entrenar: la misma idea con la compuerta fija.
- relleno: cada hueco toma la profundidad del píxel válido más cercano y se usa E2 («sin» queda sin profundidad).

`umbral` y k se eligen en la validación por escenas, con los modelos de validación (val_E1, val_E2) y la media de
todas las condiciones; el test se evalúa una sola vez, con las 3 semillas. Resultado en
runs/rgbd/combinaciones/resultado.json.

    python src/combinar_e1_e2.py

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
from entrenar_rgbd import DEV, IGNORAR, RUNS, cargar, lote_a_gpu, nombre_corrida, normalizar_rgb  # noqa: E402

E1, E2 = ("E1", 1e-3), ("E2", 3e-4)          # lr elegidas en la validación por escenas (DECISIONES.md)
UMBRALES = (0.3, 0.5, 0.7, 0.9)
VENTANAS = (1, 3, 7, 15, 31)


def rellenar(prof, valida):
    """Cada hueco toma la profundidad del píxel válido más cercano (distancia euclídea)."""
    if valida.all() or not valida.any():
        return prof
    _, (iy, ix) = ndimage.distance_transform_edt(~valida, return_indices=True)
    return prof[iy, ix]


def mezcla(p1, p2, valida, k):
    m = F.adaptive_avg_pool2d(valida.float(), p1.shape[-2:])
    if k > 1:
        m = F.avg_pool2d(m, k, 1, k // 2, count_include_pad=False)
    return (1 - m) * p1 + m * p2


def recorrer(m1, m2, datos, posiciones, lote=4):
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
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p1 = m1(x).float().softmax(1)
                p2 = m2(x, prof, valida).float().softmax(1)
                p2r = m2(x, prof_r, prof_r > 0).float().softmax(1)
            sumar(cond, "E1", p1, et)
            sumar(cond, "E2", p2, et)
            sumar(cond, "relleno", p2r, et)
            frac = valida.float().mean((1, 2, 3)).view(-1, 1, 1, 1)
            for u in UMBRALES:
                sumar(cond, f"interruptor_{u:g}", torch.where(frac >= u, p2, p1), et)
            for v in VENTANAS:
                sumar(cond, f"mezcla_{v}", mezcla(p1, p2, valida, v), et)
    res = {}
    for (cond, var), c in conf.items():
        c = c.double()
        iou = c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)
        res.setdefault(cond, {})[var] = float(100 * iou.mean())
    return res


def main():
    tr = nyuv2.cache("entrenamiento")
    _, pos_val = nyuv2.validacion_por_escenas()
    m1, _ = cargar(RUNS / nombre_corrida(*E1, 0, validacion=True))
    m2, _ = cargar(RUNS / nombre_corrida(*E2, 0, validacion=True))
    val = recorrer(m1, m2, tr, pos_val)
    del m1, m2
    media = lambda var: float(np.mean([val[c][var] for c in rgbd.CONDICIONES]))  # noqa: E731
    umbral = max(UMBRALES, key=lambda u: media(f"interruptor_{u:g}"))
    ventana = max(VENTANAS, key=lambda v: media(f"mezcla_{v}"))
    print("validación (media de las condiciones): "
          + ", ".join(f"{v} {media(v):.2f}" for v in val[rgbd.CONDICIONES[0]]), flush=True)
    print(f"elegidos: umbral {umbral:g}, ventana {ventana}", flush=True)

    test = nyuv2.cache("test")
    variantes = ["E1", "E2", f"interruptor_{umbral:g}", f"mezcla_{ventana}", "relleno"]
    por_semilla = []
    for s in range(3):
        m1, _ = cargar(RUNS / nombre_corrida(*E1, s))
        m2, _ = cargar(RUNS / nombre_corrida(*E2, s))
        por_semilla.append(recorrer(m1, m2, test, np.arange(len(test["rgb"]))))
        del m1, m2
        torch.cuda.empty_cache()
        print(f"semilla {s} evaluada", flush=True)
    tabla = {c: {v: [r[c][v] for r in por_semilla] for v in variantes} for c in rgbd.CONDICIONES}
    out = {"validacion": val, "umbral": umbral, "ventana": ventana, "variantes": variantes,
           "test": {c: {v: {"media": st.mean(x), "de": st.stdev(x), "semillas": x} for v, x in fila.items()}
                    for c, fila in tabla.items()}}
    carpeta = RUNS / "combinaciones"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "resultado.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'condición':14s}" + "".join(f"{v:>16s}" for v in variantes))
    for c, fila in tabla.items():
        print(f"{c:14s}" + "".join(f"{st.mean(x):10.2f} ±{st.stdev(x):4.2f}" for x in fila.values()))


if __name__ == "__main__":
    main()
