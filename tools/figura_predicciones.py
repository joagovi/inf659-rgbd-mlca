# -*- coding: utf-8 -*-
"""Qué predicen los modelos: entrada (color, profundidad cruda), etiqueta y la segmentación de cada modelo con
profundidad completa, cruda y sin profundidad, en unas pocas imágenes del test.

    python tools/figura_predicciones.py                 # elige 3 imágenes y escribe ES y EN
    python tools/figura_predicciones.py --imagenes 12 300 517

Necesita NYUv2 (`datos/nyuv2`), los pesos preentrenados y los `cabeza.pt` de `runs/rgbd/` (semilla 0). Solo
evalúa: no entrena ni cambia ningún resultado. Salidas: `notebooks/figs/12_2_predicciones_<i>.png` (español) y
`proposal/figs/predicciones_<i>_en.png` (inglés).

Elección por defecto: entre las imágenes del test con al menos un 20 % de huecos en la cruda (píxeles
etiquetados), las de los percentiles 50, 75 y 90 del IoU por imagen de E11r con la cruda (semilla 0, de su
`por_imagen.npz`): una típica, una buena y una muy buena. Son ejemplos, no el promedio (ver el mIoU de
cada modelo en `ARQUITECTURAS.md`).

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import entrenar_rgbd as E  # noqa: E402
import nyuv2  # noqa: E402
import rgbd  # noqa: E402

MODELOS = [("E1u", 0.001), ("E11", 0.003), ("E11d", 0.001), ("E11r", 0.001), ("E4ed", 0.001)]
CONDICIONES = ("completa", "cruda", "sin")
PALETA = np.vstack([plt.cm.tab20(np.arange(20))[:, :3], plt.cm.tab20b(np.arange(20))[:, :3]])   # 40 colores
TXT = {
    "es": {"color": "color", "cruda": "profundidad cruda (blanco = hueco, {:.0f} %)", "etiqueta": "etiqueta (40 clases)",
           "cond": {"completa": "profundidad completa", "cruda": "profundidad cruda", "sin": "sin profundidad"},
           "que": {"E1u": "E1u · solo color", "E11": "E11 · suma, sin dropout", "E11d": "E11d · suma + dropout",
                   "E11r": "E11r · suma + dropout, cruda", "E4ed": "E4ed · MLCA + dropout"},
           "acierto": "{:.0f} % de píxeles", "sin_etq": "sin etiqueta",
           "titulo": "Imagen {} del test de NYUv2 · semilla 0 · % de píxeles etiquetados bien clasificados"},
    "en": {"color": "colour", "cruda": "raw depth (white = hole, {:.0f} %)", "etiqueta": "ground truth (40 classes)",
           "cond": {"completa": "full depth", "cruda": "raw depth", "sin": "no depth"},
           "que": {"E1u": "E1u · RGB only", "E11": "E11 · sum, no dropout", "E11d": "E11d · sum + dropout",
                   "E11r": "E11r · sum + dropout, raw", "E4ed": "E4ed · MLCA + dropout"},
           "acierto": "{:.0f} % of pixels", "sin_etq": "unlabelled",
           "titulo": "NYUv2 test image {} · seed 0 · % of labelled pixels classified correctly"},
}


def elegir(percentiles=(50, 75, 90), minimo_huecos=0.20):
    """Imágenes con al menos `minimo_huecos` de huecos en la cruda (píxeles etiquetados), en los percentiles dados
    del IoU por imagen de E11r con la cruda (semilla 0): una típica, una buena y una muy buena."""
    datos = nyuv2.cache("test")
    z = np.load(E.RUNS / "E11r_lr0.001_s0" / "por_imagen.npz")
    iou = z["cruda_inter"].sum(1) / np.maximum(z["cruda_union"].sum(1), 1)
    etq = np.asarray(datos["etiqueta"]) > 0
    huecos = ((np.asarray(datos["cruda"]) == 0) & etq).sum((1, 2)) / np.maximum(etq.sum((1, 2)), 1)
    cand = np.where(huecos >= minimo_huecos)[0]
    orden = cand[np.argsort(iou[cand])]
    return [int(orden[min(len(orden) - 1, int(p / 100 * len(orden)))]) for p in percentiles]


def predecir(imagenes):
    datos = nyuv2.cache("test")
    pred = {}
    for exp, lr in MODELOS:
        modelo, _ = E.cargar(E.RUNS / f"{exp}_lr{lr:g}_s0")
        modelo.to(E.DEV).eval()
        for i in imagenes:
            rgb, _, et = E.lote_a_gpu(datos, np.array([i]))
            for cond in CONDICIONES:
                p, v = rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i))
                prof = torch.from_numpy(p).to(E.DEV)[None, None]
                valida = torch.from_numpy(v).to(E.DEV)[None, None]
                with torch.no_grad(), E.autocast():
                    lg = modelo(E.normalizar_rgb(rgb), prof, valida)
                pred[exp, i, cond] = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear",
                                                   align_corners=False).argmax(1)[0].cpu().numpy()
        del modelo
        torch.cuda.empty_cache()
    return datos, pred


def pintar(clases, ignorar_negro=True):
    img = PALETA[np.clip(clases, 0, 39)]
    img[clases == E.IGNORAR] = 0.0 if ignorar_negro else 1.0
    return img


def figura(datos, pred, i, idioma, nombres):
    t = TXT[idioma]
    et = datos["etiqueta"][i].astype(np.int64)
    et = np.where(et == 0, E.IGNORAR, et - 1)
    cruda = datos["cruda"][i].astype(np.float32) / 1e3
    hueco = (cruda == 0)
    fig, ejes = plt.subplots(1 + len(MODELOS), 3, figsize=(9.6, 2.45 * (1 + len(MODELOS)) + 0.9), facecolor="white")
    ejes[0, 0].imshow(datos["rgb"][i])
    ejes[0, 0].set_title(t["color"], fontsize=10)
    c = np.ma.masked_where(hueco, cruda)
    ejes[0, 1].imshow(c, cmap="viridis")
    ejes[0, 1].set_title(t["cruda"].format(100 * hueco[et != E.IGNORAR].mean()), fontsize=10)
    ejes[0, 2].imshow(pintar(et))
    ejes[0, 2].set_title(t["etiqueta"], fontsize=10)
    m = et != E.IGNORAR
    for f, (exp, _) in enumerate(MODELOS, 1):
        for col, cond in enumerate(CONDICIONES):
            p = pred[exp, i, cond]
            ax = ejes[f, col]
            ax.imshow(pintar(np.where(m, p, E.IGNORAR)))
            ax.set_title(f"{t['cond'][cond]} · " + t["acierto"].format(100 * (p[m] == et[m]).mean()), fontsize=9)
            if col == 0:
                ax.set_ylabel(t["que"][exp], fontsize=10, fontweight="bold")
    for ax in ejes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(False)
    presentes = [k for k in np.unique(et) if k != E.IGNORAR]
    manijas = [Patch(color=PALETA[k], label=nombres[k + 1]) for k in presentes] + [Patch(color="black",
                                                                                          label=t["sin_etq"])]
    fig.legend(handles=manijas, loc="lower center", ncol=min(6, len(manijas)), fontsize=8, frameon=False)
    fig.suptitle(t["titulo"].format(i), fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.035 + 0.012 * (len(manijas) // 6), 1, 0.985))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenes", type=int, nargs="*")
    a = ap.parse_args()
    imagenes = a.imagenes or elegir()
    print("imágenes del test:", imagenes)
    datos, pred = predecir(imagenes)
    nombres = nyuv2.clases40()
    for i in imagenes:
        for idioma, ruta in (("es", RAIZ / "notebooks" / "figs" / f"12_2_predicciones_{i}.png"),
                             ("en", RAIZ / "proposal" / "figs" / f"predicciones_{i}_en.png")):
            fig = figura(datos, pred, i, idioma, nombres)
            fig.savefig(ruta, dpi=130, facecolor="white")
            plt.close(fig)
            print(ruta.relative_to(RAIZ))


if __name__ == "__main__":
    main()
