# -*- coding: utf-8 -*-
"""NYUv2 etiquetado (1 449 imágenes) leído directamente del .mat oficial, sin extraer copias al disco.

Archivos en `datos/nyuv2/` (ignorado por git), descargados de sus fuentes:
- `nyu_depth_v2_labeled.mat` (2.97 GB): http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- `splits.mat` (795 entrenamiento / 654 test): http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
- `classMapping40.mat` (894 -> 40 clases, Gupta et al. 2013): github.com/ankurhanda/nyuv2-meta-data (idéntico
  byte a byte al de VainF/nyuv2-python-toolkit).

La lectura sigue la preparación de NYUv2 de los autores de ESANet/EMSANet (TUI-NICR/nicr-scene-analysis-datasets,
Apache 2.0): ejes del .mat transpuestos, índices de `splits.mat` desde 1, clase 0 = sin etiqueta.

`rawDepths` es la profundidad del Kinect ya proyectada al plano de la imagen de color, con 0 donde no hay medida;
`depths` es la misma tras rellenar los huecos (colorización de Levin et al.). Uso de IA: escrito con asistencia
de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "datos" / "nyuv2"
# Recorte de Eigen et al. (2014): fuera queda el borde blanco que deja el registro del Kinect
EIGEN = (slice(45, 471), slice(41, 601))


def clases40():
    """Nombres de las 40 clases (índices 1..40; el 0 es «sin etiqueta»)."""
    m = loadmat(DIR / "classMapping40.mat")
    return ["sin etiqueta"] + [str(x[0]) for x in m["className"][0]]


def indices(split):
    """Índices (desde 0) del split oficial: 'entrenamiento' (795) o 'test' (654)."""
    s = loadmat(DIR / "splits.mat")
    return s["trainNdxs" if split == "entrenamiento" else "testNdxs"][:, 0] - 1


def escenas(split):
    """Nombre de la escena de cada imagen del split (p. ej. 'kitchen_0004'). Los splits oficiales no comparten
    escenas (249 de entrenamiento, 215 de test)."""
    with h5py.File(DIR / "nyu_depth_v2_labeled.mat", "r") as f:
        refs = f["scenes"][0]
        return [''.join(chr(c) for c in f[refs[i]][:, 0]) for i in indices(split)]


def validacion_por_escenas(fraccion=0.15, semilla=0):
    """Separa del entrenamiento una validación por escenas enteras (las imágenes de una escena se parecen).
    Devuelve posiciones dentro del split de entrenamiento: (entrenamiento, validación)."""
    esc = np.array(escenas("entrenamiento"))
    unicas = np.unique(esc)
    val = np.random.default_rng(semilla).choice(unicas, round(fraccion * len(unicas)), replace=False)
    es_val = np.isin(esc, val)
    return np.flatnonzero(~es_val), np.flatnonzero(es_val)


class NYUv2:
    """Acceso por índice al .mat: devuelve rgb uint8 (480,640,3), profundidad rellenada y cruda en metros
    float32 (480,640; la cruda con 0 en los huecos) y etiqueta de 40 clases uint8 (0 = sin etiqueta)."""

    def __init__(self, split="test"):
        self.idx = indices(split)
        self.mapa = np.concatenate([[0], loadmat(DIR / "classMapping40.mat")["mapClass"][0]]).astype(np.uint8)
        self._f = None

    def __len__(self):
        return len(self.idx)

    def _abrir(self):
        if self._f is None:                       # se abre al primer uso: así cada worker tiene su manejador
            self._f = h5py.File(DIR / "nyu_depth_v2_labeled.mat", "r")
        return self._f

    def __getitem__(self, i):
        f, j = self._abrir(), int(self.idx[i])
        rgb = np.asarray(f["images"][j]).transpose(2, 1, 0)                 # (3,W,H) -> (H,W,3)
        prof = np.asarray(f["depths"][j], dtype=np.float32).T
        cruda = np.asarray(f["rawDepths"][j], dtype=np.float32).T
        etiqueta = self.mapa[np.asarray(f["labels"][j]).T]
        return {"indice": j, "rgb": rgb, "profundidad": prof, "cruda": cruda, "etiqueta": etiqueta}


def cache(split):
    """Copia sin comprimir del split en `datos/nyuv2/cache/`, leída como memmap (sin cargarla en RAM).

    El .mat va comprimido con gzip y leer una imagen al azar cuesta ~26 ms; la copia se hace una vez (~3 GB los
    dos splits). Profundidades en milímetros como uint16, como en la preparación de ESANet/EMSANet.
    Devuelve un dict de arrays: rgb (N,480,640,3) uint8, prof y cruda (N,480,640) uint16 en mm, etiqueta uint8.
    """
    d = DIR / "cache"
    campos = {"rgb": ((480, 640, 3), np.uint8), "prof": ((480, 640), np.uint16), "cruda": ((480, 640), np.uint16),
              "etiqueta": ((480, 640), np.uint8)}
    hecho = d / f"{split}.ok"
    if not hecho.exists():
        d.mkdir(exist_ok=True)
        ds = NYUv2(split)
        out = {k: np.lib.format.open_memmap(d / f"{split}_{k}.npy", "w+", t, (len(ds),) + s)
               for k, (s, t) in campos.items()}
        for i in range(len(ds)):
            s = ds[i]
            out["rgb"][i] = s["rgb"]
            out["prof"][i] = np.round(s["profundidad"] * 1e3)
            out["cruda"][i] = np.round(s["cruda"] * 1e3)
            out["etiqueta"][i] = s["etiqueta"]
        for a in out.values():
            a.flush()
        hecho.write_text("ok")
    return {k: np.load(d / f"{split}_{k}.npy", mmap_mode="r") for k in campos}
