# -*- coding: utf-8 -*-
"""Catálogo de los experimentos: qué cambia en cada uno (del registro) y su mIoU en el test (de sus resultado.json).

    python tools/catalogo_experimentos.py        # reescribe la tabla de ARQUITECTURAS.md entre sus dos marcas

La configuración sale de `entrenar_rgbd.EXPERIMENTOS`; las cifras, de `estadistica.cargar` (media de las 3
semillas con la lr elegida en la validación), las mismas que `auditoria/05_tablas_estadisticas.md`. Un
experimento con menos de 3 semillas terminadas sale como «en curso» o «en cola»; uno que no está en ninguna cola,
«no se corrió». El esquema (A-E) es el panel de `tools/figura_arquitecturas.py`.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import estadistica  # noqa: E402
from entrenar_rgbd import EXPERIMENTOS, RUNS  # noqa: E402

DOC = RAIZ / "ARQUITECTURAS.md"
INICIO, FIN = "<!-- catálogo: inicio -->", "<!-- catálogo: fin -->"
DECISION = {"E1": "v3", "E2": "v3", "E4": "v3", "E5": "v3", "E1u": "D-03", "E7": "D-03", "E8": "D-03", "E2h": "D-03",
            "E2u": "D-04", "E9": "D-04", "E10": "D-05", "E11": "D-05"}
CONDICIONES = ("completa", "cruda", "aleatoria_50", "sin")
MASCARA = {"binaria": "binaria", "ninguna": "sin máscara", "fraccional": "fraccional"}


def decision(e, c):
    return DECISION.get(e) or (re.search(r"D-0\d", c.familia) or ["D-06"])[0]


def esquema(c):
    if c.aux:
        return "D"
    if c.fusion in ("rgb", "respaldo"):
        return {"rgb": "A", "respaldo": "E"}[c.fusion]
    return "C" if c.codificador else "B"


def cabeza(c):
    return {"rgb": "—", "mlca": f"MLCA {MASCARA[c.mascara]}", "respaldo": "suma + experto de color"}.get(c.fusion,
                                                                                                         "suma")


def codificador(c):
    if not c.codificador:
        return "—"
    return f"MLCA {MASCARA[c.mascara]}" if c.codificador == "mlca" else "suma (W′·P4)"


def al_entrenar(c):
    if c.aux:
        return "rellenada, como objetivo"
    if c.fusion == "rgb":
        return "—"
    partes = ["cruda" if c.huecos == "cruda" else "rellenada"]
    if c.huecos is True:
        partes.append("huecos simulados")
    if c.dropout:
        partes.append(f"dropout {c.dropout:g}")
    return " + ".join(partes)


def resultados(e, c):
    """(lr, {condición: media}, gemelo del que se deriva o None), o (None, texto del estado, None)."""
    r = estadistica.cargar(RUNS, e)
    if r is None or len(r["semillas"]) < 3:
        n = 0 if r is None else len(r["semillas"])
        if n:
            return None, f"en curso ({n}/3 semillas)", None
        return None, "en cola" if "D-08" in c.familia else "no se corrió", None
    medias = {k: st.mean(x[k]["miou"] for x in r["res"]) for k in CONDICIONES}
    gemelo = None
    if r["derivado"]:
        d = json.loads((r["carpetas"][0] / "resultado.json").read_text(encoding="utf-8"))["derivado_de"]
        gemelo = d.split("_lr")[0]
    return r["lr"], medias, gemelo


def lineas_registro():
    """Línea de cada experimento en `EXPERIMENTOS`, para enlazarlo desde la tabla."""
    texto = (RAIZ / "src" / "entrenar_rgbd.py").read_text(encoding="utf-8").splitlines()
    return {m.group(1): i for i, l in enumerate(texto, 1) if (m := re.match(r'\s+"(E\w+)": Exp\(', l))}


def tabla():
    lineas = lineas_registro()
    filas = ["| Exp. | Dec. | Esq. | Qué es | Cabeza | Codificador | Rasgos de prof. | Etapa 4 | Profundidad al "
             "entrenar | lr | Completa | Cruda | 50 %* | Sin |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, c in EXPERIMENTOS.items():
        lr, medias, gemelo = resultados(e, c)
        nombre = f"[{e}](src/entrenar_rgbd.py#L{lineas[e]})" + ("†" if gemelo else "")
        que = c.que + (f" (derivado de {gemelo})" if gemelo else "")
        rasgos = "—" if c.fusion == "rgb" else ("C3·P4·C5" if c.rasgos == "c3p4c5" else "P3·P4·P5")
        etapa4 = "ajustada" if "stage4" in c.entrenable else "congelada"
        if lr is None:
            cifras = f"{medias} | | | |"
            lr_txt = "—"
        else:
            cifras = " | ".join(f"{medias[k]:.2f}" for k in CONDICIONES)
            lr_txt = f"{lr:g}"
        filas.append(f"| {nombre} | {decision(e, c)} | {esquema(c)} | {que} | {cabeza(c)} | {codificador(c)} | "
                     f"{rasgos} | {etapa4} | {al_entrenar(c)} | {lr_txt} | {cifras} |")
    notas = [
        "",
        "mIoU en el test oficial de NYUv2 (654 imágenes, 40 clases), media de 3 semillas con la lr elegida en la "
        "validación por escenas. Fuente: los `resultado.json` de `runs/rgbd/`, leídos con `estadistica.cargar`; "
        "las mismas cifras, con desviación, IC y cada semilla, están en `auditoria/05_tablas_estadisticas.md`. "
        "Las de la v3 y D-03 a D-05 son exploratorias (el test se usó en esas decisiones).",
        "",
        "\\* 50 %: la mitad de los píxeles con profundidad, borrados al azar. Sin: toda la profundidad a cero.",
        "",
        "† Derivado sin entrenar: con la profundidad rellenada y sin dropout, la máscara no actúa al entrenar, así "
        "que es la misma red que su gemelo, evaluada con otra máscara (D-06).",
        "",
        "Esq.: el panel de las figuras (A-E). Dec.: la decisión de `DECISIONES.md` que lo introdujo (v3: el diseño "
        "de la propuesta, antes de D-01). El nombre de cada experimento enlaza a su configuración en el registro "
        "`EXPERIMENTOS` de `src/entrenar_rgbd.py`.",
    ]
    return "\n".join(filas + notas)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    texto = DOC.read_text(encoding="utf-8")
    a, b = texto.index(INICIO) + len(INICIO), texto.index(FIN)
    DOC.write_text(texto[:a] + "\n" + tabla() + "\n" + texto[b:], encoding="utf-8")
    print(f"{DOC.relative_to(RAIZ)}: {len(EXPERIMENTOS)} experimentos")


if __name__ == "__main__":
    main()
