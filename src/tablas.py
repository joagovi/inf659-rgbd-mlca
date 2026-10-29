# -*- coding: utf-8 -*-
"""Tablas de resultados en markdown, solo a partir de los archivos de runs/rgbd (sin GPU ni pesos):

- cada experimento con la lr elegida en la validación por escenas: media ± desviación de las 3 semillas en el
  test, en cada condición de la profundidad, más la media y la peor de las condiciones;
- las pruebas sin entrenar de `combinar_e1_e2.py` (D-03) y de `combinar.py` (D-04 en adelante).

    python src/tablas.py
    python src/tablas.py --validacion      # además, el barrido de lr de cada experimento

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_rgbd import FUSION, RUNS, elegida, nombre_corrida  # noqa: E402
from rgbd import CONDICIONES  # noqa: E402

COLUMNAS = ["completa", "cruda", ("aleatoria_10", "aleatoria_30", "aleatoria_50"), "regiones", "sin",
            ("ruido_10cm", "ruido_20cm", "ruido_50cm")]
TITULOS = ["completa", "cruda", "aleatoria 10 / 30 / 50 %", "regiones", "sin", "ruido 10 / 20 / 50 cm"]


def fila(nombre, por_cond):
    """por_cond: {condición: [mIoU por semilla]}; la desviación solo en «completa» y «sin», como en DECISIONES."""
    celdas = []
    for col in COLUMNAS:
        if isinstance(col, tuple):
            celdas.append(" / ".join(f"{st.mean(por_cond[c]):.2f}" for c in col))
        elif col in ("completa", "sin") and len(por_cond[col]) > 1:
            celdas.append(f"{st.mean(por_cond[col]):.2f} ± {st.stdev(por_cond[col]):.2f}")
        else:
            celdas.append(f"{st.mean(por_cond[col]):.2f}")
    medias = [st.mean(por_cond[c]) for c in CONDICIONES]
    return f"| {nombre} | " + " | ".join(celdas) + f" | {st.mean(medias):.2f} | {min(medias):.2f} |"


def cabecera(primera):
    return (f"| {primera} | " + " | ".join(TITULOS) + " | media | peor |\n"
            + "|---" * (len(TITULOS) + 3) + "|")


def experimentos(validacion):
    print(cabecera("Experimento (lr)"))
    barridos = []
    for exp in FUSION:
        try:
            (lr, aux), val = elegida(exp)
        except FileNotFoundError:
            continue
        rs = [RUNS / nombre_corrida(exp, lr, s, aux=aux) / "resultado.json" for s in range(3)]
        rs = [json.loads(r.read_text(encoding="utf-8")) for r in rs if r.exists()]
        if not rs:
            continue
        por_cond = {c: [r["resultados"][c]["miou"] for r in rs] for c in CONDICIONES}
        nombre = f"{exp} ({lr:g}{'' if aux is None else f', aux {aux:g}'}){'' if len(rs) == 3 else f', {len(rs)} semillas'}"
        print(fila(nombre, por_cond))
        barridos.append(f"{exp}: " + ", ".join(f"{l:g}{'' if a is None else f' aux {a:g}'} → {v:.2f}"
                                               for (l, a), v in sorted(val.items(), key=str)))
    if validacion:
        print("\nValidación por escenas (profundidad completa, semilla 0):\n")
        print("\n".join(f"- {b}" for b in barridos))


def combinaciones():
    archivos = sorted((RUNS / "combinaciones").glob("**/resultado.json"))
    for archivo in archivos:
        r = json.loads(archivo.read_text(encoding="utf-8"))
        if "color" in r:                                    # combinar.py
            titulo = f"{r['color'][0]} ({r['color'][1]:g}) + {r['fusion'][0]} ({r['fusion'][1]:g})"
            elegido = f"umbral {r['umbral']:g}, {r['mezcla']}, {r['relleno']}"
        else:                                               # combinar_e1_e2.py (D-03)
            titulo, elegido = "E1 + E2 (D-03)", f"umbral {r['umbral']:g}, ventana {r['ventana']}"
        print(f"\n{titulo}, elegido en la validación: {elegido}\n")
        print(cabecera("Variante"))
        for v in r["variantes"]:
            print(fila(v, {c: r["test"][c][v]["semillas"] for c in CONDICIONES}))
        if "ensamble_color" in r:
            e = r["ensamble_color"]
            print(f"\nControl, dos modelos de color de semillas distintas: {e['media']:.2f} ± {e['de']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validacion", action="store_true")
    a = ap.parse_args()
    print("Test oficial (654 imágenes), 3 semillas, mIoU de 40 clases. Los modelos sin profundidad dan lo mismo "
          "en todas las condiciones.\n")
    experimentos(a.validacion)
    combinaciones()


if __name__ == "__main__":
    main()
