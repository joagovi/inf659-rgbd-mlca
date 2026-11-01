# -*- coding: utf-8 -*-
"""Criterio de D-06 (auditoria/06_plan_experimentos.md) sobre los por_imagen.npz, sin GPU.

    python tools/criterio_d06.py

«Aporta» si la diferencia es > 0, su IC por remuestreo de imágenes excluye el 0 y el signo se repite en las 3
semillas; «no aporta» si |diferencia| < 0.3 y el IC incluye el 0; si no, «no concluyente». Se añade
«perjudica» (lo simétrico de «aporta») para no confundirlo con «no concluyente». Holm sobre las tres condiciones
principales de cada contraste: el IC de cada condición usa el α de su posición en el orden de Holm.
Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver DECISIONES.md.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import estadistica as E  # noqa: E402

RUNS = RAIZ / "runs" / "rgbd"
CONDS = ("cruda", "regiones", "sin")
CONTRASTES = (("E4c", "E3c"), ("E4cd", "E3cd"), ("E3cd", "E3c"), ("E4cd", "E4c"), ("E3c", "E2c"))


def boot(a, b, cond, B=4000):
    def leer(c):
        z = np.load(c / "por_imagen.npz")
        k = cond if f"{cond}_inter" in z else "completa"
        return z[f"{k}_inter"].astype(np.int64), z[f"{k}_union"].astype(np.int64)
    A, Bm = [leer(c) for c in a["carpetas"]], [leer(c) for c in b["carpetas"]]
    n = A[0][0].shape[0]
    rng = np.random.default_rng(0)
    d = np.array([np.mean([E.miou_de(x[i], y[i]) for x, y in A]) - np.mean([E.miou_de(x[i], y[i]) for x, y in Bm])
                  for i in (rng.integers(0, n, n) for _ in range(B))])
    obs = np.mean([E.miou_de(x, y) for x, y in A]) - np.mean([E.miou_de(x, y) for x, y in Bm])
    return obs, d, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def main():
    for a_, b_ in CONTRASTES:
        a, b = E.cargar(RUNS, a_), E.cargar(RUNS, b_)
        if a is None or b is None:
            print(f"{a_} − {b_}: faltan corridas")
            continue
        filas = []
        for cond in CONDS:
            obs, d, p = boot(a, b, cond)
            ps = [ra[cond]["miou"] - rb.get(cond, rb["completa"])["miou"] for ra, rb in zip(a["res"], b["res"])]
            filas.append({"cond": cond, "dif": obs, "semillas": ps, "p": p, "d": d})
        orden = sorted(range(len(filas)), key=lambda k: filas[k]["p"])
        p_holm, acum = {}, 0.0
        for r, k in enumerate(orden):
            acum = max(acum, min(1.0, (len(filas) - r) * filas[k]["p"]))
            p_holm[k] = acum
        print(f"\n{a_} − {b_}")
        for k, fi in enumerate(filas):
            alfa = 0.05 / (len(filas) - orden.index(k))
            lo, hi = np.percentile(fi["d"], [100 * alfa / 2, 100 * (1 - alfa / 2)])
            if fi["dif"] > 0 and lo > 0 and all(x > 0 for x in fi["semillas"]) and p_holm[k] < 0.05:
                v = "aporta"
            elif fi["dif"] < 0 and hi < 0 and all(x < 0 for x in fi["semillas"]) and p_holm[k] < 0.05:
                v = "perjudica"
            elif abs(fi["dif"]) < 0.3 and lo <= 0 <= hi:
                v = "no aporta"
            else:
                v = "no concluyente"
            print(f"  {fi['cond']:9s} {fi['dif']:+.2f}  IC (Holm) [{lo:+.2f}, {hi:+.2f}]  semillas "
                  + " ".join(f"{x:+.2f}" for x in fi["semillas"]) + f"  p Holm {p_holm[k]:.4f}  → {v}")


if __name__ == "__main__":
    main()
