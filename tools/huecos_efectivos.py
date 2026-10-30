# -*- coding: utf-8 -*-
"""Porcentajes nominales y efectivos de pérdida de profundidad en cada condición de evaluación (D-06).

    python tools/huecos_efectivos.py                 # sin datos: supone `depths` sin ceros (se comprueba con --nyuv2)
    python tools/huecos_efectivos.py --nyuv2         # con NYUv2: también «cruda» y los ceros de `depths`

Para cada condición y cada imagen del test (las mismas semillas que la evaluación: el índice de la imagen):
fracción de píxeles sin profundidad en la imagen completa y en el recorte de Eigen, y, a la resolución de los
rasgos (1/8, 1/16, 1/32), la fracción media válida por celda (lo que usa la máscara fraccional), la proporción de
celdas sin ningún píxel válido (las únicas que la máscara fraccional quita) y la proporción de celdas que quita la
máscara binaria (fracción válida ≤ 0.5).

La profundidad rellenada (`depths`) no es una medida del sensor: en el test, el 30.7 % de sus píxeles (12.3 % en
el recorte de Eigen) son interpolados por la colorización de Levin et al. (D-01). «completa» trata esos píxeles
como válidos.

Resultado: runs/degradaciones/huecos_efectivos.json y .md. Uso de IA: escrito con asistencia de IA (Claude,
Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import rgbd  # noqa: E402

NOMINAL = {"completa": 0, "aleatoria_10": 10, "aleatoria_30": 30, "aleatoria_50": 50, "sin": 100,
           "regiones": None, "cruda": None, "ruido_10cm": 0, "ruido_20cm": 0, "ruido_50cm": 0,
           "regiones_025": None, "regiones_075": None, "regiones_100": None}
ESCALAS = {"1/8": 8, "1/16": 16, "1/32": 32}
EIGEN = (slice(45, 471), slice(41, 601))


def celdas(valida, s):
    h, w = valida.shape
    return valida.reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def medir(n, prof=None, cruda=None, h=480, w=640, umbral=0.5):
    res = {}
    for cond in rgbd.CONDICIONES + rgbd.CONDICIONES_EXTRA:
        if cond == "cruda" and cruda is None:
            continue
        filas = []
        for i in range(n):
            p = prof[i] if prof is not None else np.full((h, w), 2500, np.uint16)
            c = cruda[i] if cruda is not None else p
            _, v = rgbd.condicion(cond, p, c, semilla=i)
            f = {"imagen": 1 - v.mean(), "eigen": 1 - v[EIGEN].mean()}
            for e, s in ESCALAS.items():
                m = celdas(v.astype(np.float32), s)
                f[f"m_medio_{e}"] = m.mean()
                f[f"celdas_vacias_{e}"] = (m == 0).mean()
                f[f"celdas_quitadas_binaria_{e}"] = (m <= umbral).mean()
            filas.append(f)
        res[cond] = {k: {"media": float(np.mean([f[k] for f in filas])), "min": float(np.min([f[k] for f in filas])),
                         "max": float(np.max([f[k] for f in filas]))} for k in filas[0]}
        res[cond]["nominal_%"] = NOMINAL[cond]
    return res


def informe(res, nota):
    L = ["# Pérdida de profundidad nominal y efectiva por condición", "", nota, "",
         "| condición | nominal % | sin medida, imagen % | recorte de Eigen % | m medio 1/8 · 1/16 · 1/32 | "
         "celdas vacías % 1/8 · 1/16 · 1/32 | celdas que quita la binaria % 1/8 · 1/16 · 1/32 |", "|---" * 7 + "|"]
    for c, r in res.items():
        nom = "—" if r["nominal_%"] is None else r["nominal_%"]
        tr = lambda k: " · ".join(f"{100 * r[f'{k}_{e}']['media']:.1f}" for e in ESCALAS)  # noqa: E731
        L.append(f"| {c} | {nom} | {100 * r['imagen']['media']:.2f} ({100 * r['imagen']['min']:.1f}–"
                 f"{100 * r['imagen']['max']:.1f}) | {100 * r['eigen']['media']:.2f} | "
                 + " · ".join(f"{r[f'm_medio_{e}']['media']:.3f}" for e in ESCALAS)
                 + f" | {tr('celdas_vacias')} | {tr('celdas_quitadas_binaria')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyuv2", action="store_true")
    ap.add_argument("--n", type=int, default=654)
    a = ap.parse_args()
    out = {}
    if a.nyuv2:
        import nyuv2
        te, tr = nyuv2.cache("test"), nyuv2.cache("entrenamiento")
        out["ceros_en_depths"] = {s: float(np.mean([(d["prof"][i] == 0).mean() for i in range(len(d["prof"]))]))
                                  for s, d in (("test", te), ("entrenamiento", tr))}
        res = medir(len(te["prof"]), te["prof"], te["cruda"])
        nota = "Con NYUv2 (test, 654 imágenes)."
    else:
        res = medir(a.n)
        nota = ("Sin NYUv2: supone que `depths` no tiene ceros (se comprueba con --nyuv2) y no incluye «cruda». "
                f"{a.n} imágenes, semillas 0..{a.n - 1} como en la evaluación.")
    out.update({"nota": nota, "condiciones": res})
    sal = RAIZ / "runs" / "degradaciones"
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "huecos_efectivos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (sal / "huecos_efectivos.md").write_text(informe(res, nota), encoding="utf-8")
    print(informe(res, nota))


if __name__ == "__main__":
    main()
