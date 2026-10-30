# -*- coding: utf-8 -*-
"""Tablas estadísticas a partir de los resultado.json (y, si existen, de los por_imagen.npz), sin GPU (D-06).

    python src/estadistica.py                    # escribe auditoria/05_tablas_estadisticas.md y runs/estadistica.json
    python src/estadistica.py --runs <carpeta>

- Media ± desviación de las 3 semillas e intervalo t al 95 % (n = 3, t = 4.303): con tres semillas el intervalo
  es ancho; es la incertidumbre por la semilla, no por las imágenes.
- Caída respecto a la profundidad completa, por semilla.
- Diferencias pareadas por semilla (A − B con la misma semilla). El emparejamiento solo comparte de verdad la
  inicialización y el orden de los datos cuando las dos configuraciones consumen el generador igual; en la v3,
  E5 (dropout con el generador de los datos) y los experimentos con huecos no lo hacen.
- Con por_imagen.npz: intervalo por remuestreo de imágenes (bootstrap pareado, las mismas imágenes para A y B y
  para las tres semillas), que mide la variabilidad entre imágenes del test.
- IoU por clase.
- Los controles de ensamble usan pares de semillas solapados (0-1, 1-2, 0-2): se dan los tres valores y el rango.

Todo resultado sobre el test oficial obtenido hasta D-05 es **exploratorio**: el equipo usó el test para los
diagnósticos de D-03 a D-05 y para decidir los experimentos siguientes. Solo serán confirmatorios los de D-06
cuyas hipótesis se registren antes de evaluar el test, y una sola vez.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_rgbd import EXPERIMENTOS, RUNS, nombre_corrida  # noqa: E402
from rgbd import CONDICIONES, CONDICIONES_EXTRA  # noqa: E402

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
PRINCIPAL = ["E1", "E2", "E4", "E5"]
FACTORIAL = {"sin dropout": ("E3c", "E4c"), "con dropout": ("E3cd", "E4cd")}
PARES = [("E2", "E1", "añadir profundidad (P3-P5)"), ("E4", "E2", "MLCA frente a la suma"),
         ("E4", "E1", "MLCA frente a solo color"), ("E5", "E4", "dropout sobre MLCA (generador compartido)"),
         ("E5", "E1", "MLCA + dropout frente a solo color"), ("E2u", "E1u", "profundidad sobre el ajuste"),
         ("E9", "E1u", "respaldo con compuerta frente a E1u"),
         ("E2c", "E1", "profundidad, rasgos corregidos"), ("E4c", "E3c", "efecto de la máscara sin dropout"),
         ("E4cd", "E3cd", "efecto de la máscara con dropout"), ("E3cd", "E3c", "efecto del dropout sin máscara"),
         ("E4cd", "E4c", "efecto del dropout con máscara"), ("E4cf", "E4c", "máscara fraccional frente a binaria"),
         ("E4cf", "E4", "rasgos corregidos, misma máscara fraccional")]


def lr_elegida(runs, exp):
    val = {}
    for r in runs.glob(f"val_{exp}_lr*_s0/resultado.json"):
        o = json.loads(r.read_text(encoding="utf-8"))
        if o["exp"] == exp:
            val[(o["lr"], o.get("aux"))] = o["resultados"]["completa"]["miou"]
    claves = sorted(val, key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    return (max(claves, key=val.get), val) if val else (None, {})


def cargar(runs, exp):
    (clave, val) = lr_elegida(runs, exp)
    if clave is None:
        return None
    lr, aux = clave
    rs = []
    for s in range(3):
        p = runs / nombre_corrida(exp, lr, s, aux=aux) / "resultado.json"
        if p.exists():
            rs.append((s, json.loads(p.read_text(encoding="utf-8")), p.parent))
    if not rs:
        return None
    return {"exp": exp, "lr": lr, "aux": aux, "validacion": val, "semillas": [s for s, _, _ in rs],
            "res": [r["resultados"] for _, r, _ in rs], "carpetas": [c for _, _, c in rs],
            "clases": rs[0][1]["clases"], "derivado": any("derivado_de" in r for _, r, _ in rs)}


def resumen(x):
    n = len(x)
    m = st.mean(x)
    de = st.stdev(x) if n > 1 else float("nan")
    ic = T95.get(n, float("nan")) * de / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "media": m, "de": de, "ic95": ic, "valores": x}


def miou_de(inter, union):
    """mIoU de un conjunto de imágenes a partir de intersección y unión por imagen y clase (N,40)."""
    return 100 * float(np.mean(inter.sum(0) / np.maximum(union.sum(0), 1)))


def bootstrap_pareado(a_carpetas, b_carpetas, cond, B=2000, semilla=0):
    """IC 95 % de mIoU(A) − mIoU(B) remuestreando imágenes; promedio de semillas dentro de cada remuestreo."""
    def leer(c):
        z = np.load(c / "por_imagen.npz")
        k = cond if f"{cond}_inter" in z else "completa"            # modelos sin profundidad: una condición
        return z[f"{k}_inter"].astype(np.int64), z[f"{k}_union"].astype(np.int64)
    A, Bm = [leer(c) for c in a_carpetas], [leer(c) for c in b_carpetas]
    n = A[0][0].shape[0]
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        da = np.mean([miou_de(x[i], y[i]) for x, y in A])
        db = np.mean([miou_de(x[i], y[i]) for x, y in Bm])
        difs.append(da - db)
    obs = np.mean([miou_de(x, y) for x, y in A]) - np.mean([miou_de(x, y) for x, y in Bm])
    return {"dif": float(obs), "ic95": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))], "B": B}


def f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def tablas(runs):
    exps = {e: cargar(runs, e) for e in EXPERIMENTOS}
    exps = {e: v for e, v in exps.items() if v}
    out = {"experimentos": {}, "pares": [], "nota": "exploratorio: test usado en D-03 a D-05"}
    L = ["# Tablas estadísticas (generadas por `src/estadistica.py`)", "",
         "Calculadas solo a partir de los `resultado.json` de `runs/rgbd/`. Test oficial de NYUv2 (654 imágenes), "
         "mIoU de 40 clases. **Todas las cifras de test de la v3 son exploratorias**: el test se usó en los "
         "diagnósticos y en las decisiones de D-03 a D-05. Con tres semillas, el IC t al 95 % multiplica la "
         "desviación por 4.303/√3 ≈ 2.48.", ""]
    conds = CONDICIONES + CONDICIONES_EXTRA
    # 1. por experimento y condición
    L += ["## 1. mIoU por experimento y condición: media ± DE [IC 95 %] (semillas)", ""]
    for e, v in exps.items():
        cs = [c for c in conds if all(c in r for r in v["res"])]
        aux = "" if v["aux"] is None else f", aux {v['aux']:g}"
        der = " · derivado sin entrenar" if v["derivado"] else ""
        L += [f"### {e} · {EXPERIMENTOS[e].que} · lr {v['lr']:g}{aux} · {len(v['res'])} semillas{der}", "",
              "| condición | media ± DE | IC 95 % | por semilla | caída frente a completa (media ± DE) |",
              "|---|---|---|---|---|"]
        out["experimentos"][e] = {"lr": v["lr"], "condiciones": {}}
        base = [r["completa"]["miou"] for r in v["res"]]
        for c in cs:
            x = [r[c]["miou"] for r in v["res"]]
            s = resumen(x)
            caida = resumen([b - y for b, y in zip(base, x)])
            out["experimentos"][e]["condiciones"][c] = {**s, "caida": caida}
            L.append(f"| {c} | {f(s['media'])} ± {f(s['de'])} | ±{f(s['ic95'])} | "
                     f"{' / '.join(f(y) for y in x)} | {f(caida['media'])} ± {f(caida['de'])} |")
        L.append("")
    # 2. diferencias pareadas
    L += ["## 2. Diferencias pareadas por semilla (A − B), media ± DE [IC 95 %]", "",
          "El emparejamiento por semilla comparte la inicialización de la cabeza; el orden de los datos solo si las "
          "dos configuraciones consumen el generador igual (no en E5 frente a E4 de la v3). Las filas sin datos se "
          "omiten.", "", "| A − B | qué mide | completa | cruda | regiones | aleatoria 50 % | sin | ruido 50 cm |",
          "|---|---|---|---|---|---|---|---|"]
    for a, b, que in PARES:
        if a not in exps or b not in exps:
            continue
        sa = dict(zip(exps[a]["semillas"], exps[a]["res"]))
        sb = dict(zip(exps[b]["semillas"], exps[b]["res"]))
        comunes = sorted(set(sa) & set(sb))
        celdas = []
        for c in ("completa", "cruda", "regiones", "aleatoria_50", "sin", "ruido_50cm"):
            d = [sa[s][c]["miou"] - sb[s][c]["miou"] for s in comunes]
            r = resumen(d)
            out["pares"].append({"a": a, "b": b, "condicion": c, **r})
            celdas.append(f"{r['media']:+.2f} ± {f(r['de'])} [±{f(r['ic95'])}]")
        L.append(f"| {a} − {b} | {que} | " + " | ".join(celdas) + " |")
    # 2b. interacción del factorial
    L += ["", "## 3. Factorial 2×2 de E5 (máscara × dropout, rasgos corregidos)", ""]
    if all(x in exps for par in FACTORIAL.values() for x in par):
        L += ["| condición | máscara sin dropout (E4c − E3c) | máscara con dropout (E4cd − E3cd) | "
              "dropout sin máscara (E3cd − E3c) | interacción |", "|---|---|---|---|---|"]
        for c in conds:
            try:
                g = {k: dict(zip(exps[k]["semillas"], [r[c]["miou"] for r in exps[k]["res"]]))
                     for k in ("E3c", "E4c", "E3cd", "E4cd")}
            except KeyError:
                continue
            ss = sorted(set.intersection(*[set(v) for v in g.values()]))
            m0 = resumen([g["E4c"][s] - g["E3c"][s] for s in ss])
            m1 = resumen([g["E4cd"][s] - g["E3cd"][s] for s in ss])
            d0 = resumen([g["E3cd"][s] - g["E3c"][s] for s in ss])
            it = resumen([(g["E4cd"][s] - g["E3cd"][s]) - (g["E4c"][s] - g["E3c"][s]) for s in ss])
            L.append(f"| {c} | " + " | ".join(f"{r['media']:+.2f} ± {f(r['de'])}" for r in (m0, m1, d0, it)) + " |")
    else:
        L.append("Pendiente: faltan " + ", ".join(x for par in FACTORIAL.values() for x in par if x not in exps)
                 + " (no se han entrenado). No se rellena nada.")
    # 3. bootstrap por imágenes
    L += ["", "## 4. Variabilidad entre imágenes (bootstrap pareado de imágenes)", ""]
    hechos = 0
    for a, b, que in PARES:
        if a in exps and b in exps and all((c / "por_imagen.npz").exists()
                                           for c in exps[a]["carpetas"] + exps[b]["carpetas"]):
            for c in ("completa", "regiones", "cruda", "sin"):
                r = bootstrap_pareado(exps[a]["carpetas"], exps[b]["carpetas"], c)
                L.append(f"- {a} − {b}, {c}: {r['dif']:+.2f} [IC 95 % {r['ic95'][0]:+.2f}, {r['ic95'][1]:+.2f}]")
                hechos += 1
    if not hechos:
        L.append("No disponible: las corridas de la v3 no guardaron predicciones por imagen. Las corridas nuevas "
                 "guardan `por_imagen.npz` y esta sección se rellena sola.")
    # 4. por clase
    if all(e in exps for e in PRINCIPAL):
        L += ["", "## 5. IoU por clase, profundidad completa (media de las semillas)", "",
              "| clase | E1 | E2 | E4 | E5 | E4 − E2 | E4 sin profundidad |", "|---|---|---|---|---|---|---|"]
        iou = {e: np.mean([r["completa"]["iou_clase"] for r in exps[e]["res"]], 0) for e in PRINCIPAL}
        sin4 = np.mean([r["sin"]["iou_clase"] for r in exps["E4"]["res"]], 0)
        for i, cl in enumerate(exps["E1"]["clases"]):
            L.append(f"| {cl} | " + " | ".join(f"{iou[e][i]:.1f}" for e in PRINCIPAL)
                     + f" | {iou['E4'][i] - iou['E2'][i]:+.1f} | {sin4[i]:.1f} |")
        out["por_clase"] = {e: [float(x) for x in v] for e, v in iou.items()}
    # 5. validación
    L += ["", "## 6. Barridos de lr en la validación por escenas (profundidad completa, semilla 0)", "",
          "| exp | lr → mIoU | elegida | ¿en un extremo? |", "|---|---|---|---|"]
    for e, v in exps.items():
        claves = sorted(v["validacion"], key=str)
        if len({l for l, _ in claves}) > 1:                     # se barrió la lr
            barr, eleg = sorted(l for l, _ in claves), v["lr"]
        else:                                                   # E7: se barrió el peso auxiliar
            barr, eleg = sorted(x for _, x in claves), v["aux"]
        ext = len(barr) > 1 and eleg in (barr[0], barr[-1])
        L.append(f"| {e} | " + ", ".join(f"{l:g}{'' if x is None else f'/aux {x:g}'} → {m:.2f}"
                                         for (l, x), m in sorted(v["validacion"].items(), key=str))
                 + f" | {v['lr']:g} | {'sí' if ext else 'no'} |")
    # 6. combinaciones
    L += ["", "## 7. Combinaciones sin entrenar (complementarias, no forman parte de E1-E5)", ""]
    for arch in sorted((runs / "combinaciones").glob("**/resultado*.json")):
        r = json.loads(arch.read_text(encoding="utf-8"))
        if "color" not in r:
            titulo, sel = "E1 + E2 (D-03)", f"umbral {r['umbral']:g}, ventana {r['ventana']}"
        else:
            titulo = f"{r['color'][0]} + {r['fusion'][0]}"
            sel = f"{r['mezcla']}, {r['relleno']}, umbral {r['umbral']:g}"
        L += [f"**{titulo}** (`{arch.relative_to(runs)}`; interpolación {r.get('interpolar', 'probabilidades')}; "
              f"elegido en la validación: {sel})", "",
              "| variante | completa | regiones | sin |", "|---|---|---|---|"]
        for v_ in r["variantes"]:
            L.append(f"| {v_} | " + " | ".join(
                f"{r['test'][c][v_]['media']:.2f} ± {r['test'][c][v_]['de']:.2f}" for c in ("completa", "regiones", "sin"))
                + " |")
        if "ensamble_color" in r:
            e = r["ensamble_color"]
            L.append(f"\nControl, dos modelos de color: pares {', '.join(f'{x:.2f}' for x in e['miou'])} "
                     f"(rango {min(e['miou']):.2f}–{max(e['miou']):.2f}; pares solapados, no independientes).")
        L.append("")
    return "\n".join(L) + "\n", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--salida", default=str(Path(__file__).resolve().parents[1] / "auditoria" / "05_tablas_estadisticas.md"))
    a = ap.parse_args()
    md, js = tablas(Path(a.runs))
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(md, encoding="utf-8")
    (Path(a.runs).parent / "estadistica.json").write_text(json.dumps(js, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
