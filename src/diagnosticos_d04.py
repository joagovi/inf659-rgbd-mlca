# -*- coding: utf-8 -*-
"""Diagnósticos de los resultados de D-04, sin reentrenar. Son exploratorios: nada de lo que aquí se mide sirve
para elegir un modelo.

1. E9 por dentro: la mezcla, cada experto por separado, el α medio de la compuerta y, con los mismos dos expertos,
   la mezcla con α fijo = a · m_15 (la regla de `combinar.py`). Separa lo que aporta el experto de fusión de lo que
   aporta la compuerta aprendida. En el test (3 semillas), en la validación por escenas (las 3 lr del barrido) y
   en las imágenes de entrenamiento, que es donde se ajusta la compuerta.
2. Escalas de la fusión en E2 y E2u: mIoU con profundidad completa quitando la suma W·f_prof de una escala, o
   dejando solo una, y el tamaño relativo ‖W·f_prof‖ / ‖f_color‖ en cada una. Quitar la suma también quita su
   magnitud; por eso, además, la profundidad de una escala se cambia por la de otra imagen del test (una
   permutación fija sin puntos fijos): la suma conserva su tamaño y pierde la información de la imagen.
3. Por clase (de los resultado.json, sin GPU): ganancia de la profundidad con el codificador congelado (E2 − E1) y
   con la última etapa ajustada (E2u − E1u), y de E9 sobre su experto de color (E9 − E1u).

    python src/diagnosticos_d04.py

Resultado en runs/rgbd/diagnosticos_d04.json. Si ya existe, solo se calcula lo que le falte.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
from combinar import TOPES, fraccion  # noqa: E402
from diagnosticos_d03 import miou, sumar  # noqa: E402
from entrenar_rgbd import DEV, RUNS, cargar, lote_a_gpu, nombre_corrida, normalizar_rgb  # noqa: E402

LR = {"E1": 1e-3, "E1u": 1e-3, "E2": 3e-4, "E2u": 3e-3, "E9": 3e-4}    # elegidas en la validación (DECISIONES.md)
CONDICIONES = ("completa", "cruda", "regiones")
QUITAR = {"todas": (), "sin_1/8": ("stage2",), "sin_1/16": ("stage3",), "sin_1/32": ("stage4",),
          "solo_1/8": ("stage3", "stage4"), "solo_1/16": ("stage2", "stage4"), "solo_1/32": ("stage2", "stage3"),
          "ninguna": ("stage2", "stage3", "stage4")}
ESCALA = {"stage2": "1/8", "stage3": "1/16", "stage4": "1/32"}


def lotes(datos, posiciones, cond, lote=4):
    for k in range(0, len(posiciones), lote):
        idx = np.sort(posiciones[k:k + lote])
        rgb, _, et = lote_a_gpu(datos, idx)
        pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in idx]
        prof = torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None]
        valida = torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None]
        yield normalizar_rgb(rgb), prof, valida, et


def por_dentro_e9(carpeta, datos, posiciones, condiciones):
    modelo, _ = cargar(carpeta)
    modelo.eval()
    out = {}
    for cond in condiciones:
        variantes = ["mezcla", "color", "fusion"] + [f"tope_{a:g}" for a in TOPES]
        conf = {v: torch.zeros(40, 40, dtype=torch.long, device=DEV) for v in variantes}
        alfa, n = 0.0, 0
        for x, prof, valida, et in lotes(datos, posiciones, cond):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                sal = modelo(x, prof, valida, todo=True)
            p1, p2 = sal["rgb"].float().softmax(1), sal["fus"].float().softmax(1)
            m = fraccion(valida, p1.shape[-2:], 15)
            sumar(conf["mezcla"], sal["seg"], et)
            sumar(conf["color"], p1, et)
            sumar(conf["fusion"], p2, et)
            for a in TOPES:
                sumar(conf[f"tope_{a:g}"], (1 - a * m) * p1 + a * m * p2, et)
            alfa += float(sal["alfa"].float().sum())
            n += sal["alfa"].numel()
        out[cond] = {**{v: miou(c) for v, c in conf.items()}, "alfa_medio": alfa / n}
        print(f"{carpeta.name} {cond}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[cond].items()), flush=True)
    del modelo
    torch.cuda.empty_cache()
    return out


def escalas(carpeta, datos):
    """La cabeza se evalúa con la suma W·f_prof quitada en las escalas de cada variante de `QUITAR`."""
    modelo, _ = cargar(carpeta)
    modelo.eval()
    assert not modelo.codificador, "la última etapa del color se calcula aquí sin la profundidad"
    conf = {v: torch.zeros(40, 40, dtype=torch.long, device=DEV) for v in QUITAR}
    rel = {c: [] for c, _ in modelo.pares}
    for x, prof, valida, et in lotes(datos, np.arange(len(datos["rgb"])), "completa"):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = modelo.color(x)
            fd = rgbd.rasgos_defm(modelo.defm, prof, valida, modelo.rasgos_prof)
            suma = {c: modelo.fus[c](fd[d].to(f[c].dtype)) for c, d in modelo.pares}
            for c in suma:
                rel[c] += (suma[c].float().flatten(1).norm(dim=1) / f[c].float().flatten(1).norm(dim=1)).tolist()
            for v, quitar in QUITAR.items():
                g = {**f, **{c: f[c] + s for c, s in suma.items() if c not in quitar}}
                sumar(conf[v], rgbd.salida(modelo.evit.head, g), et)
    out = {**{v: miou(c) for v, c in conf.items()},
           **{f"relativo_{ESCALA[c]}": float(np.mean(r)) for c, r in rel.items()}}
    print(f"{carpeta.name}: " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    del modelo
    torch.cuda.empty_cache()
    return out


def permutadas(carpeta, datos, lote=4):
    """Como `escalas`, pero la suma de las escalas de cada variante usa la profundidad de otra imagen."""
    modelo, _ = cargar(carpeta)
    modelo.eval()
    assert not modelo.codificador, "la última etapa del color se calcula aquí sin la profundidad"
    n = len(datos["rgb"])
    orden = np.random.default_rng(0).permutation(n)
    otra = np.empty(n, int)
    otra[orden] = np.roll(orden, -1)                      # un solo ciclo: ninguna imagen se queda con la suya
    variantes = {"propia": (), "otra_1/8": ("stage2",), "otra_1/16": ("stage3",), "otra_1/32": ("stage4",),
                 "otra_todas": ("stage2", "stage3", "stage4")}
    conf = {v: torch.zeros(40, 40, dtype=torch.long, device=DEV) for v in variantes}

    def profundidad(posiciones):
        pv = [rgbd.condicion("completa", datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in posiciones]
        return (torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None],
                torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None])

    for k in range(0, n, lote):
        idx = np.arange(k, min(k + lote, n))
        rgb, _, et = lote_a_gpu(datos, idx)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = modelo.color(normalizar_rgb(rgb))
            fd = {"propia": rgbd.rasgos_defm(modelo.defm, *profundidad(idx), modelo.rasgos_prof),
                  "otra": rgbd.rasgos_defm(modelo.defm, *profundidad(otra[idx]), modelo.rasgos_prof)}
            suma = {q: {c: modelo.fus[c](fd[q][d].to(f[c].dtype)) for c, d in modelo.pares} for q in fd}
            for v, cambiar in variantes.items():
                g = {**f, **{c: f[c] + suma["otra" if c in cambiar else "propia"][c] for c in suma["propia"]}}
                sumar(conf[v], rgbd.salida(modelo.evit.head, g), et)
    out = {v: miou(c) for v, c in conf.items()}
    print(f"{carpeta.name}: " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    del modelo
    torch.cuda.empty_cache()
    return out


def por_clase():
    iou, clases = {}, None
    for exp in ("E1", "E1u", "E2", "E2u", "E9"):
        rs = [json.loads((RUNS / nombre_corrida(exp, LR[exp], s) / "resultado.json").read_text(encoding="utf-8"))
              for s in range(3)]
        iou[exp] = np.mean([r["resultados"]["completa"]["iou_clase"] for r in rs], 0)
        clases = rs[0]["clases"]
    gan = {"E2-E1": iou["E2"] - iou["E1"], "E2u-E1u": iou["E2u"] - iou["E1u"], "E9-E1u": iou["E9"] - iou["E1u"],
           "E1u-E1": iou["E1u"] - iou["E1"]}
    corr = {"E2-E1_vs_E2u-E1u": float(np.corrcoef(gan["E2-E1"], gan["E2u-E1u"])[0, 1]),
            "E2u-E1u_vs_E9-E1u": float(np.corrcoef(gan["E2u-E1u"], gan["E9-E1u"])[0, 1]),
            "E2u-E1u_vs_E1u-E1": float(np.corrcoef(gan["E2u-E1u"], gan["E1u-E1"])[0, 1])}
    print("correlaciones por clase: " + "  ".join(f"{k} {v:.2f}" for k, v in corr.items()), flush=True)
    return {"clases": clases, "correlaciones": corr,
            **{f"ganancia_{k}": [round(float(x), 2) for x in v] for k, v in gan.items()}}


def main():
    test, tr = nyuv2.cache("test"), nyuv2.cache("entrenamiento")
    _, pos_val = nyuv2.validacion_por_escenas()
    archivo = RUNS / "diagnosticos_d04.json"
    out = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {
        "nota": "exploratorio, sin reentrenar; no sirve para elegir modelos (D-04)"}
    pasos = {
        "por_clase": por_clase,
        "e9_test": lambda: [por_dentro_e9(RUNS / nombre_corrida("E9", LR["E9"], s), test, np.arange(len(test["rgb"])),
                                          CONDICIONES) for s in range(3)],
        "e9_validacion": lambda: {f"{lr:g}": por_dentro_e9(RUNS / nombre_corrida("E9", lr, 0, validacion=True), tr,
                                                           pos_val, ("completa",)) for lr in (3e-4, 1e-3, 3e-3)},
        "escalas": lambda: {exp: [escalas(RUNS / nombre_corrida(exp, LR[exp], s), test) for s in range(3)]
                            for exp in ("E2", "E2u")},
        # la compuerta se ajusta en estas imágenes, que el experto de color (E1u) ya vio durante 60 épocas
        "e9_entrenamiento": lambda: [por_dentro_e9(RUNS / nombre_corrida("E9", LR["E9"], s), tr,
                                                   np.arange(len(tr["rgb"])), ("completa",)) for s in range(3)],
        "permutadas": lambda: {exp: [permutadas(RUNS / nombre_corrida(exp, LR[exp], s), test) for s in range(3)]
                               for exp in ("E2", "E2u")},
    }
    for clave, paso in pasos.items():
        if clave not in out:
            out[clave] = paso()
            archivo.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
