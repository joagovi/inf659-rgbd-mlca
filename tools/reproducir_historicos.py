# -*- coding: utf-8 -*-
"""Antes de lanzar D-06: el código nuevo reproduce los resultados guardados de la v3 (como en D-03 a D-05).

    python tools/reproducir_historicos.py            # E1u, E2u (sus pesos vienen en el ZIP de la v3) y lo que haya

Recarga cada modelo con cabeza.pt, lo evalúa con profundidad completa y sin profundidad (semilla 0) y compara con
su resultado.json. Falla (código 1) si alguna diferencia supera `--tolerancia` puntos de mIoU. Con
`--sin-mascara` evalúa además los E4 que tengan pesos con la máscara quitada (el E3 de la v3, D-06), sin guardar.
Necesita NYUv2 y los pesos oficiales. Uso de IA: escrito con asistencia de IA (Claude, Anthropic).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import entrenar_rgbd as E  # noqa: E402
import nyuv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerancia", type=float, default=0.05)
    ap.add_argument("--sin-mascara", action="store_true")
    a = ap.parse_args()
    test = nyuv2.cache("test")
    pos = np.arange(len(test["rgb"]))
    malos, informe = 0, {}
    for carpeta in sorted(E.RUNS.glob("E*_s0")):
        if not (carpeta / "cabeza.pt").exists():
            continue
        guardado = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
        modelo, _ = E.cargar(carpeta)
        nuevo = E.evaluar(modelo, test, pos, ["completa", "sin"])
        difs = {c: nuevo[c]["miou"] - guardado["resultados"][c]["miou"] for c in nuevo}
        informe[carpeta.name] = {"guardado": {c: guardado["resultados"][c]["miou"] for c in nuevo},
                                 "ahora": {c: nuevo[c]["miou"] for c in nuevo}, "diferencia": difs}
        ok = all(abs(d) <= a.tolerancia for d in difs.values())
        malos += not ok
        print(f"{carpeta.name}: " + "  ".join(f"{c} {nuevo[c]['miou']:.4f} ({d:+.4f})" for c, d in difs.items())
              + ("" if ok else "  <- NO REPRODUCE"), flush=True)
        if a.sin_mascara and modelo.fusion == "mlca":
            modelo.mascara = "ninguna"
            r = E.evaluar(modelo, test, pos, ["completa", "sin", "cruda", "regiones"])
            informe[carpeta.name]["sin_mascara"] = {c: r[c]["miou"] for c in r}
            print(f"  sin máscara (exploratorio): " + "  ".join(f"{c} {r[c]['miou']:.2f}" for c in r), flush=True)
    (E.RUNS / "reproduccion_historicos.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    return 1 if malos else 0


if __name__ == "__main__":
    sys.exit(main())
