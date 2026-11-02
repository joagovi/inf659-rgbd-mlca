# -*- coding: utf-8 -*-
"""Esquemas de las arquitecturas probadas, en español y en inglés.

    python tools/figura_arquitecturas.py      # proposal/figs/arquitecturas_{principales,complementarias}_{es,en}.{png,pdf}

Principales: A solo color (E1, E1u); B fusión en la cabeza (E2, E2u, E10, la familia «c»); C fusión también a la
entrada de la última etapa del color (E11 y sus variantes de D-07 y D-08). Debajo, los dos operadores de fusión:
la suma y la MLCA. Complementarias: D profundidad como maestra (E7) y E dos expertos con compuerta (E8, E9).

Se dibujan a partir de `rgbd.RGBD` y `mlca.MLCA`, sin importarlos. Qué experimento es cada uno, con su
configuración y sus resultados, está en `ARQUITECTURAS.md`. Uso de IA: escrito con asistencia de IA (Claude,
Anthropic); ver `DECISIONES.md`.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "proposal" / "figs"
TINTA, GRIS = "#1d2430", "#5b6472"
CONGELADO = "#e4e7ec"
SUMA, SUMA_F = "#2a78d6", "#dce9fb"      # los mismos tonos que la figura de robustez (cuaderno 12)
MLCA, MLCA_F = "#eb6834", "#fde3d6"
PULG = 0.48                               # pulgadas por unidad de dibujo
LEYENDA = {"congelado": (CONGELADO, TINTA, "-"), "entrena": ("white", TINTA, "-"), "variable": ("white", TINTA, "--"),
           "solo_ent": ("white", TINTA, ":"), "suma": (SUMA_F, SUMA, "-"), "mlca": (MLCA_F, MLCA, "-")}

# Columnas y filas de cada panel (unidades de dibujo; y hacia arriba)
XC, XD, XF = 2.55, 7.6, 5.3               # color, profundidad, fusión de la cabeza
WC, WD = 3.9, 3.5
Y_IN, Y_E13, Y_FP, Y_E4, Y_FUS, Y_CAB, Y_SEG = 15.3, 13.1, 10.95, 9.0, 6.7, 4.6, 2.7
H_E4 = 1.5
DEFM_Y, DEFM_H = 11.75, 3.6

TXT = {
    "es": {
        "A": "A · Solo color", "B": "B · Fusión en la cabeza", "C": "C · Fusión también en el codificador",
        "A_exp": "E1: se entrena solo la cabeza\nE1u: también la etapa 4",
        "B_exp": "suma: E2, E2h, E2u, E10, E2c, E2cd\nMLCA: E4, E5, E3c, E3cd, E4c, E4cd, E4cf, E4cfd",
        "C_exp": "suma: E11, E11d, E11r\nMLCA: E3e, E4e, E4ed, E3er, E4er",
        "rgb": "Imagen RGB", "prof": "Profundidad",
        "evit_sub": "ADE20K · tallo y etapas 1–3\ncongeladas",
        "e4": "Etapa 4 · 1/32",
        "e4_A": "congelada: E1\najustada: E1u (lr × 0.1)", "e4_B": "congelada: v3 y familia «c»\najustada: E2u, E10",
        "e4_C": "ajustada · lr × 0.1",
        "defm_sub": "congelado\npreentrenado\nen profundidad",
        "sin_prof": "no usa\nprofundidad",
        "cab": "Cabeza de EfficientViT-Seg", "cab_sub": "se entrena · lee 1/8, 1/16 y 1/32",
        "seg": "Segmentación · 40 clases",
        "suma": "suma", "mlca": "MLCA", "escalas": "fusión en 1/8, 1/16 y 1/32",
        "c3p4c5": "C3 · P4 · C5", "p3p4p5": "P3 · P4 · P5", "v3": "E2, E2h, E2u, E4, E5:\nP3 · P4 · P5",
        "fp": "F′", "p4": "P4 · 1/16", "cab_sumas": "en la cabeza, siempre sumas",
        "leyenda": {"congelado": "congelado", "entrena": "se entrena",
                    "variable": "congelada o ajustada, según el experimento", "solo_ent": "solo al entrenar",
                    "suma": "suma", "mlca": "MLCA"},
        "op_suma": "Suma", "op_mlca": "atención cruzada lineal enmascarada",
        "suma_l": ("W: convolución 1×1 sin sesgo, empieza en 0 (g = f)",
                   "cada posición recibe la profundidad de su posición",
                   "no usa la máscara: un hueco entra como profundidad 0"),
        "mlca_l": ("q del color; k y v de la profundidad; $W_o$ empieza en 0",
                   "todas las posiciones reciben el mismo resumen S",
                   "de la imagen entera, sin información de posición",
                   "m = validez: binaria (E4) o 1 en todas (E3)",
                   "sin profundidad: o = 0 y g = f"),
        "color": "color", "profundidad": "profundidad", "hueco": "hueco",
        "D": "D · Profundidad como maestra", "E": "E · Dos expertos y compuerta",
        "D_exp": "E7: E1u + cabeza auxiliar de profundidad\n(peso 1, 3 o 10; se elige en validación)",
        "E_exp": "E8: se entrena con huecos simulados\nE9: igual, con el experto de color de E1u",
        "prof_obj": "Profundidad", "prof_obj_sub": "solo como objetivo",
        "cab_seg": "Cabeza de segmentación", "cab_seg_sub": "se entrena",
        "cab_aux": "Cabeza de profundidad", "cab_aux_sub": "auxiliar",
        "perdida": "Pérdida auxiliar", "perdida_sub": "invariante a la escala\n(Eigen et al., 2014)",
        "solo_ent": "la profundidad solo\nsupervisa al entrenar;\nal evaluar no se usa",
        "e4_E": "congelada: E8\najustada: E9",
        "cab_col": "Cabeza de color", "cab_col_sub": "experto de color\n(E9: el de E1u)",
        "cab_fus": "Cabeza de fusión", "cab_fus_sub": "experto RGB-D",
        "comp": "Compuerta α", "comp_sub": "sin medida\ncerca: α = 0",
        "validez": "validez", "mezcla_sub": "por píxel",
    },
    "en": {
        "A": "A · Colour only", "B": "B · Fusion at the head", "C": "C · Fusion also in the encoder",
        "A_exp": "E1: only the head is trained\nE1u: stage 4 as well",
        "B_exp": "sum: E2, E2h, E2u, E10, E2c, E2cd\nMLCA: E4, E5, E3c, E3cd, E4c, E4cd, E4cf, E4cfd",
        "C_exp": "sum: E11, E11d, E11r\nMLCA: E3e, E4e, E4ed, E3er, E4er",
        "rgb": "RGB image", "prof": "Depth",
        "evit_sub": "ADE20K · stem, stages 1–3\nfrozen",
        "e4": "Stage 4 · 1/32",
        "e4_A": "frozen: E1\nfine-tuned: E1u (lr × 0.1)", "e4_B": "frozen: v3 and «c» family\nfine-tuned: E2u, E10",
        "e4_C": "fine-tuned · lr × 0.1",
        "defm_sub": "frozen\npretrained\non depth",
        "sin_prof": "no depth\ninput",
        "cab": "EfficientViT-Seg head", "cab_sub": "trained · reads 1/8, 1/16 and 1/32",
        "seg": "Segmentation · 40 classes",
        "suma": "sum", "mlca": "MLCA", "escalas": "fusion at 1/8, 1/16 and 1/32",
        "c3p4c5": "C3 · P4 · C5", "p3p4p5": "P3 · P4 · P5", "v3": "E2, E2h, E2u, E4, E5:\nP3 · P4 · P5",
        "fp": "F′", "p4": "P4 · 1/16", "cab_sumas": "at the head, always sums",
        "leyenda": {"congelado": "frozen", "entrena": "trained",
                    "variable": "frozen or fine-tuned, depending on the experiment", "solo_ent": "training only",
                    "suma": "sum", "mlca": "MLCA"},
        "op_suma": "Sum", "op_mlca": "masked linear cross-attention",
        "suma_l": ("W: 1×1 convolution without bias, starts at 0 (g = f)",
                   "each position gets the depth at its own position",
                   "no mask: a hole enters as depth 0"),
        "mlca_l": ("q from colour; k and v from depth; $W_o$ starts at 0",
                   "every position gets the same summary S",
                   "of the whole image, with no positional information",
                   "m = validity: binary (E4) or 1 everywhere (E3)",
                   "without depth: o = 0 and g = f"),
        "color": "colour", "profundidad": "depth", "hueco": "hole",
        "D": "D · Depth as a teacher", "E": "E · Two experts and a gate",
        "D_exp": "E7: E1u + auxiliary depth head\n(weight 1, 3 or 10; chosen on validation)",
        "E_exp": "E8: trained with simulated holes\nE9: the same, with E1u as the colour expert",
        "prof_obj": "Depth", "prof_obj_sub": "only as a target",
        "cab_seg": "Segmentation head", "cab_seg_sub": "trained",
        "cab_aux": "Depth head", "cab_aux_sub": "auxiliary",
        "perdida": "Auxiliary loss", "perdida_sub": "scale-invariant log depth\n(Eigen et al., 2014)",
        "solo_ent": "depth only\nsupervises training;\nnot used at test time",
        "e4_E": "frozen: E8\nfine-tuned: E9",
        "cab_col": "Colour head", "cab_col_sub": "colour expert\n(E9: E1u's)",
        "cab_fus": "Fusion head", "cab_fus_sub": "RGB-D expert",
        "comp": "Gate α", "comp_sub": "no depth\nnearby: α = 0",
        "validez": "validity", "mezcla_sub": "per pixel",
    },
}


def caja(ax, cx, cy, w, h, titulo, sub=None, fc="white", ec=TINTA, ls="-", tam=11.5, color=TINTA):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=fc, ec=ec, lw=1.8, ls=ls, zorder=2))
    if sub is None:
        ax.text(cx, cy, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color=color, zorder=3)
        return
    n = sub.count("\n") + 1
    ax.text(cx, cy + h / 2 - 0.42, titulo, ha="center", va="center", fontsize=tam, fontweight="bold", color=color,
            zorder=3)
    ax.text(cx, cy - 0.2 - 0.14 * (n - 1), sub, ha="center", va="center", fontsize=9, color=GRIS, zorder=3,
            linespacing=1.25)


def caja_doble(ax, cx, cy, w, h, t, solo_suma=False):
    """Bloque de fusión: «suma | MLCA» si el experimento elige el operador, «suma» si no."""
    if solo_suma:
        caja(ax, cx, cy, w, h, t["suma"], fc=SUMA_F, ec=SUMA, color=SUMA)
        return
    marco = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                           fc="none", ec=TINTA, lw=1.8, zorder=2.5)
    ax.add_patch(marco)
    for x0, fc in ((cx - w / 2, SUMA_F), (cx, MLCA_F)):
        r = Rectangle((x0, cy - h / 2), w / 2, h, fc=fc, ec="none", zorder=2)
        ax.add_patch(r)
        r.set_clip_path(marco)
    ax.plot([cx, cx], [cy - h / 2, cy + h / 2], color=TINTA, lw=0.9, zorder=2.6)
    ax.text(cx - w / 4, cy, t["suma"], ha="center", va="center", fontsize=11, fontweight="bold", color=SUMA, zorder=3)
    ax.text(cx + w / 4, cy, t["mlca"], ha="center", va="center", fontsize=11, fontweight="bold", color=MLCA, zorder=3)


def flecha(ax, puntos, color=TINTA, ls="-", lw=1.6, punta=True):
    """Polilínea por `puntos`, con punta de flecha en el último."""
    xs, ys = zip(*puntos)
    if not punta:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
        return
    if len(puntos) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    ax.annotate("", xy=puntos[-1], xytext=puntos[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=13, shrinkA=0, shrinkB=0))


def etiqueta(ax, x, y, s, ha="left", tam=9, color=GRIS, **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=tam, color=color, zorder=3, **kw)


def titulo(ax, t, clave, y=17.0):
    ax.text(0.15, y, t[clave], ha="left", va="center", fontsize=14, fontweight="bold", color=TINTA)
    ax.text(0.15, y - 0.82, t[clave + "_exp"], ha="left", va="center", fontsize=9.5, color=GRIS, linespacing=1.3)


def columna_color(ax, t, sub_e4, ls_e4="-", fp=False):
    """Imagen, etapas 1-3, (F′), etapa 4 y la unión de f2, f3 (bus a la izquierda) con f4 en (XC, Y_FUS)."""
    caja(ax, XC, Y_IN, WC, 0.95, t["rgb"])
    flecha(ax, [(XC, Y_IN - 0.475), (XC, Y_E13 + 0.75)])
    caja(ax, XC, Y_E13, WC, 1.5, "EfficientViT-B1", t["evit_sub"], fc=CONGELADO)
    if fp:
        caja_doble(ax, XC, Y_FP, 2.8, 0.95, t)
        etiqueta(ax, XC - 1.55, Y_FP, t["fp"], ha="right", tam=11, color=TINTA, fontstyle="italic")
        flecha(ax, [(XC, Y_E13 - 0.75), (XC, Y_FP + 0.475)])
        flecha(ax, [(XC, Y_FP - 0.475), (XC, Y_E4 + H_E4 / 2)])
    else:
        flecha(ax, [(XC, Y_E13 - 0.75), (XC, Y_E4 + H_E4 / 2)])
    caja(ax, XC, Y_E4, WC, H_E4, t["e4"], sub_e4, ls=ls_e4)
    flecha(ax, [(XC, Y_E4 - H_E4 / 2), (XC, Y_FUS)], punta=False)
    flecha(ax, [(XC - WC / 2, Y_E13), (0.25, Y_E13), (0.25, Y_FUS), (XC, Y_FUS)], punta=False)
    ax.plot([XC], [Y_FUS], "o", color=TINTA, ms=4, zorder=3)
    etiqueta(ax, 0.42, (Y_E4 - H_E4 / 2 + Y_FUS) / 2, "f₂  f₃", color=TINTA)
    etiqueta(ax, XC + 0.15, (Y_E4 - H_E4 / 2 + Y_FUS) / 2, "f₄", color=TINTA)


def columna_prof(ax, t, p4_al_codificador=False):
    caja(ax, XD, Y_IN, WD, 0.95, t["prof"])
    flecha(ax, [(XD, Y_IN - 0.475), (XD, DEFM_Y + DEFM_H / 2)])
    caja(ax, XD, DEFM_Y, WD, DEFM_H, "DeFM ResNet-18", t["defm_sub"], fc=CONGELADO)
    if p4_al_codificador:
        flecha(ax, [(XD - WD / 2, Y_FP), (XC + 1.4, Y_FP)])
        etiqueta(ax, (XD - WD / 2 + XC + 1.4) / 2, Y_FP + 0.32, t["p4"], ha="center", color=TINTA)


def cabeza_y_salida(ax, t, x_entrada):
    caja(ax, XF, Y_CAB, 8.6, 1.2, t["cab"], t["cab_sub"])
    flecha(ax, [(x_entrada, Y_FUS if x_entrada == XC else Y_FUS - 0.525), (x_entrada, Y_CAB + 0.6)])
    caja(ax, XF, Y_SEG, 5.8, 0.95, t["seg"])
    flecha(ax, [(XF, Y_CAB - 0.6), (XF, Y_SEG + 0.475)])


def fusion_cabeza(ax, t, solo_suma):
    caja_doble(ax, XF, Y_FUS, 2.9, 1.05, t, solo_suma=solo_suma)
    etiqueta(ax, XF, Y_FUS + 0.85, t["cab_sumas"] if solo_suma else t["escalas"], ha="center", tam=8.5)
    flecha(ax, [(XC, Y_FUS), (XF - 1.45, Y_FUS)])
    flecha(ax, [(XD, DEFM_Y - DEFM_H / 2), (XD, Y_FUS), (XF + 1.45, Y_FUS)])
    etiqueta(ax, XD + 0.15, 8.75, t["c3p4c5"], color=TINTA)


def panel_a(ax, t):
    titulo(ax, t, "A")
    columna_color(ax, t, t["e4_A"], ls_e4="--")
    etiqueta(ax, XD, DEFM_Y, t["sin_prof"], ha="center", tam=10.5, fontstyle="italic")
    cabeza_y_salida(ax, t, XC)


def panel_b(ax, t):
    titulo(ax, t, "B")
    columna_color(ax, t, t["e4_B"], ls_e4="--")
    columna_prof(ax, t)
    fusion_cabeza(ax, t, solo_suma=False)
    etiqueta(ax, XD + 0.15, 7.95, t["v3"], tam=7.5, linespacing=1.2)
    cabeza_y_salida(ax, t, XF)


def panel_c(ax, t):
    titulo(ax, t, "C")
    columna_color(ax, t, t["e4_C"], fp=True)
    columna_prof(ax, t, p4_al_codificador=True)
    fusion_cabeza(ax, t, solo_suma=True)
    cabeza_y_salida(ax, t, XF)


def celdas(ax, x0, y, n, w, fc, ec, hueco=None):
    for i in range(n):
        es_hueco = i == hueco
        ax.add_patch(Rectangle((x0 + i * (w + 0.25), y - w / 2), w, w, fc="white" if es_hueco else fc, ec=ec,
                               lw=1.2, hatch="////" if es_hueco else None, zorder=2))
    return [x0 + i * (w + 0.25) + w / 2 for i in range(n)]


def operadores(ax, t):
    """Los dos operadores de fusión, con un pictograma: suma uno a uno; MLCA, todo a un resumen S y de S a todos."""
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for x0, ec in ((0.15, SUMA), (15.15, MLCA)):
        ax.add_patch(FancyBboxPatch((x0, 0.1), 14.7, 5.8, boxstyle="round,pad=0,rounding_size=0.25", fc="white",
                                    ec=ec, lw=1.8, zorder=1))
    # Suma
    yf, yd, w = 4.3, 1.5, 0.8
    xs = celdas(ax, 0.9, yf, 4, w, SUMA_F, SUMA)
    celdas(ax, 0.9, yd, 4, w, CONGELADO, GRIS)
    for x in xs:
        flecha(ax, [(x, yd + w / 2), (x, yf - w / 2)], color=SUMA, lw=1.4)
    etiqueta(ax, 0.9, yf + 0.75, t["color"], tam=9)
    etiqueta(ax, 0.9, yd - 0.75, t["profundidad"], tam=9)
    ax.text(5.6, 5.25, t["op_suma"], fontsize=13, fontweight="bold", color=SUMA, va="center")
    ax.text(5.6, 4.35, r"$g = f + W\,d$", fontsize=14, color=TINTA, va="center")
    for i, linea in enumerate(t["suma_l"]):
        etiqueta(ax, 5.6, 3.25 - 0.62 * i, linea, tam=10, color=TINTA)
    # MLCA
    xs = celdas(ax, 15.9, yf, 4, w, MLCA_F, MLCA)
    xd = celdas(ax, 15.9, yd, 4, w, CONGELADO, GRIS, hueco=2)
    xs_c = 15.9 + (4 * w + 3 * 0.25) / 2
    ax.add_patch(FancyBboxPatch((xs_c - 0.4, 2.55), 0.8, 0.75, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=MLCA, ec=MLCA, zorder=2))
    ax.text(xs_c, 2.925, "S", ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=3)
    for i, x in enumerate(xd):
        if i != 2:
            flecha(ax, [(x, yd + w / 2), (xs_c + (x - xs_c) * 0.25, 2.55)], color=MLCA, lw=1.2)
    for x in xs:
        flecha(ax, [(xs_c + (x - xs_c) * 0.25, 3.3), (x, yf - w / 2)], color=MLCA, lw=1.2)
    etiqueta(ax, 15.9, yf + 0.75, t["color"], tam=9)
    etiqueta(ax, 15.9, yd - 0.75, t["profundidad"] + "  ·  m = 0: " + t["hueco"], tam=9)
    ax.text(20.6, 5.25, "MLCA", fontsize=13, fontweight="bold", color=MLCA, va="center")
    ax.text(22.3, 5.22, t["op_mlca"], fontsize=10.5, color=MLCA, va="center")
    ax.text(20.6, 4.35, r"$g = f + W_o\,o, \qquad o_i = \varphi(q_i)^\top S \;/\; (\varphi(q_i)^\top z + \varepsilon)$",
            fontsize=12, color=TINTA, va="center")
    ax.text(20.6, 3.45, r"$S = \sum_j m_j\,\varphi(k_j)\,v_j^\top,\quad z = \sum_j m_j\,\varphi(k_j),\quad"
            r"\varphi = \mathrm{ReLU}$", fontsize=11.5, color=TINTA, va="center")
    for i, linea in enumerate(t["mlca_l"]):
        etiqueta(ax, 20.6, 2.6 - 0.5 * i, linea, tam=9.5, color=TINTA)


def leyenda(fig, t, y, claves):
    manijas = [Patch(fc=LEYENDA[c][0], ec=LEYENDA[c][1], ls=LEYENDA[c][2], lw=1.5, label=t["leyenda"][c])
               for c in claves]
    fig.legend(handles=manijas, loc="center", bbox_to_anchor=(0.5, y), ncol=len(claves), frameon=False,
               fontsize=10.5, handlelength=2.2, handleheight=1.1, columnspacing=1.6)


def ejes(fig, x_pulg, y_pulg, w_unid, h_unid, ylim):
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x_pulg / W, y_pulg / H, w_unid * PULG / W, h_unid * PULG / H])
    ax.set_xlim(0, w_unid)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def principales(t):
    alto_panel = 17.6 - 1.9
    fig = plt.figure(figsize=(15.0, 0.15 + alto_panel * PULG + 0.5 + 6 * PULG + 0.1))
    y_panel = 0.1 + 6 * PULG + 0.5
    for i, dibujar in enumerate((panel_a, panel_b, panel_c)):
        dibujar(ejes(fig, 0.1 + i * 5.0, y_panel, 10, alto_panel, (1.9, 17.6)), t)
    leyenda(fig, t, (0.1 + 6 * PULG + 0.25) / fig.get_size_inches()[1],
            ("congelado", "entrena", "variable", "suma", "mlca"))
    W, H = fig.get_size_inches()
    operadores(fig.add_axes([0.3 / W, 0.1 / H, 30 * PULG / W, 6 * PULG / H]), t)
    return fig


def panel_d(ax, t):
    titulo(ax, t, "D")
    columna_color(ax, t, t["e4_C"])
    caja(ax, XD, Y_IN, WD, 0.95, t["prof_obj"], ls=":")
    etiqueta(ax, XD, Y_IN - 0.85, t["prof_obj_sub"], ha="center")
    etiqueta(ax, XD, DEFM_Y, t["solo_ent"], ha="center", tam=10, fontstyle="italic")
    caja(ax, 2.45, Y_CAB, 4.5, 1.2, t["cab_seg"], t["cab_seg_sub"], tam=10)
    flecha(ax, [(XC, Y_FUS), (XC, Y_CAB + 0.6)])
    caja(ax, 7.4, Y_CAB, 4.2, 1.2, t["cab_aux"], t["cab_aux_sub"], ls=":", tam=10)
    flecha(ax, [(XC, Y_FUS), (7.4, Y_FUS), (7.4, Y_CAB + 0.6)], ls=":")
    caja(ax, 2.45, Y_SEG, 4.6, 0.95, t["seg"], tam=10)
    flecha(ax, [(XC, Y_CAB - 0.6), (XC, Y_SEG + 0.475)])
    caja(ax, 7.4, Y_SEG, 3.9, 1.4, t["perdida"], t["perdida_sub"], ls=":", tam=10.5)
    flecha(ax, [(7.4, Y_CAB - 0.6), (7.4, Y_SEG + 0.7)], ls=":")
    flecha(ax, [(XD + WD / 2, Y_IN), (9.85, Y_IN), (9.85, Y_SEG), (7.4 + 1.95, Y_SEG)], ls=":")


def panel_e(ax, t):
    titulo(ax, t, "E")
    columna_color(ax, t, t["e4_E"], ls_e4="--")
    columna_prof(ax, t)
    xs, xm, y_mez, y_sal, xg = 6.6, 4.3, 2.3, 0.85, 8.65
    caja(ax, xs, Y_FUS, 2.5, 1.05, t["suma"], fc=SUMA_F, ec=SUMA, color=SUMA)
    etiqueta(ax, xs, Y_FUS + 0.85, t["escalas"], ha="center", tam=8.5)
    flecha(ax, [(XC, Y_FUS), (xs - 1.25, Y_FUS)])
    flecha(ax, [(8.9, DEFM_Y - DEFM_H / 2), (8.9, Y_FUS), (xs + 1.25, Y_FUS)])
    etiqueta(ax, 8.75, 8.75, t["p3p4p5"], ha="right", color=TINTA)
    caja(ax, XC, Y_CAB, 3.4, 1.4, t["cab_col"], t["cab_col_sub"], tam=10.5)
    flecha(ax, [(XC, Y_FUS), (XC, Y_CAB + 0.7)])
    caja(ax, xs, Y_CAB, 3.4, 1.4, t["cab_fus"], t["cab_fus_sub"], tam=10.5)
    flecha(ax, [(xs, Y_FUS - 0.525), (xs, Y_CAB + 0.7)])
    ax.add_patch(FancyBboxPatch((xm - 2.45, y_mez - 0.55), 4.9, 1.1, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="white", ec=TINTA, lw=1.8, zorder=2))
    ax.text(xm, y_mez + 0.15, r"$p = (1-\alpha)\,p_{\mathrm{color}} + \alpha\,p_{\mathrm{fus}}$", ha="center",
            va="center", fontsize=11.5, color=TINTA, zorder=3)
    etiqueta(ax, xm, y_mez - 0.3, t["mezcla_sub"], ha="center", tam=8.5)
    flecha(ax, [(XC, Y_CAB - 0.7), (XC, y_mez + 0.55)])
    flecha(ax, [(xs, Y_CAB - 0.7), (xs, y_mez + 0.55)])
    caja(ax, xg, y_mez, 2.5, 1.1, t["comp"], tam=10)
    etiqueta(ax, xg, y_mez - 1.05, t["comp_sub"], ha="center", tam=8, linespacing=1.2)
    flecha(ax, [(XD + WD / 2, Y_IN), (9.85, Y_IN), (9.85, y_mez + 1.15), (xg, y_mez + 1.15), (xg, y_mez + 0.55)],
           color=GRIS, lw=1.3)
    etiqueta(ax, 9.7, y_mez + 1.42, t["validez"], ha="right", tam=8.5)
    flecha(ax, [(xg - 1.25, y_mez), (xm + 2.45, y_mez)])
    caja(ax, xm, y_sal, 5.0, 0.85, t["seg"], tam=10.5)
    flecha(ax, [(xm, y_mez - 0.55), (xm, y_sal + 0.425)])


def complementarias(t):
    alto_panel = 17.6 - 0.2
    fig = plt.figure(figsize=(10.2, 0.15 + alto_panel * PULG + 0.6))
    for i, dibujar in enumerate((panel_d, panel_e)):
        dibujar(ejes(fig, 0.1 + i * 5.0, 0.6, 10, alto_panel, (0.2, 17.6)), t)
    leyenda(fig, t, 0.3 / fig.get_size_inches()[1], ("congelado", "entrena", "variable", "solo_ent", "suma"))
    return fig


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    for idioma, t in TXT.items():
        for nombre, hacer in (("principales", principales), ("complementarias", complementarias)):
            fig = hacer(t)
            for ext in ("png", "pdf"):
                ruta = SALIDA / f"arquitecturas_{nombre}_{idioma}.{ext}"
                fig.savefig(ruta, dpi=200, facecolor="white")
                print(ruta.relative_to(RAIZ))
            plt.close(fig)


if __name__ == "__main__":
    main()
