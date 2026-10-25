# -*- coding: utf-8 -*-
"""Segmentador RGB-D de la propuesta: EfficientViT-B1 (color) y DeFM ResNet-18 (profundidad), congelados.

Se entrenan solo la fusión y la cabeza de EfficientViT (preentrenada en ADE20K, con el clasificador cambiado a
40 clases). La fusión actúa en las tres escalas que lee la cabeza (1/8, 1/16, 1/32), emparejadas con P3-P5 de
DeFM (`09_mlca_bloque` §2.2):

- `rgb` (E1): sin profundidad.
- `concat` (E2): f_color + W·f_prof, con W (1×1, sin sesgo) iniciada en cero. Equivale a concatenar y proyectar
  con la parte de color fija en la identidad, así que E1 es el caso W = 0. Sin máscara: los huecos entran.
- `mlca` (E4): f_color + MLCA(f_color, f_prof, validez) (`mlca.py`).

También las condiciones de evaluación de la profundidad (propuesta §7; D-01). Uso de IA: escrito con asistencia
de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import efficientvit_seg as S
from mlca import MLCA, validez_a_rasgos

DEFM_MEAN = torch.tensor([0.248880, 0.495620, 0.492858]).view(1, 3, 1, 1)   # defm/utils/utils.py
DEFM_STD = torch.tensor([0.139357, 0.271314, 0.297177]).view(1, 3, 1, 1)
PARES = [("stage2", "P3"), ("stage3", "P4"), ("stage4", "P5")]           # color y profundidad a la misma escala


def entrada_defm(prof_m, valida):
    """Profundidad métrica (B,1,H,W) en metros y validez (B,1,H,W) bool -> entrada normalizada de DeFM (B,3,H,W).

    Es `convert_to_3channel_metric` del código oficial, en torch, con una diferencia declarada (D-01 en `DECISIONES.md`): el mínimo
    y el máximo del tercer canal se toman solo sobre los píxeles válidos. Los huecos valen 0 en los tres canales,
    como en el oficial.
    """
    d = torch.where(valida, prof_m.clamp(0, 100), torch.zeros_like(prof_m))
    lg = torch.log1p(d)
    c1 = lg / math.log1p(100.0)
    c2 = (lg / math.log1p(9.0)).clamp(0, 1)                  # 9 m y no 10: lo explica el código oficial
    inf = torch.full_like(lg, float("inf"))
    lmin = torch.where(valida, lg, inf).amin(dim=(2, 3), keepdim=True)
    lmax = torch.where(valida, lg, -inf).amax(dim=(2, 3), keepdim=True)
    rango = lmax - lmin
    ok = torch.isfinite(rango) & (rango > 0)
    c3 = torch.where(ok & valida, (lg - lmin) / torch.where(ok, rango, torch.ones_like(rango)), torch.zeros_like(lg))
    x = torch.cat([c1, c2, c3], 1)
    return (x - DEFM_MEAN.to(x.device)) / DEFM_STD.to(x.device)


def cargar_defm():
    return torch.hub.load("leggedrobotics/defm:main", "defm_resnet18", pretrained=True, trust_repo=True,
                          verbose=False)


class RGBD(nn.Module):
    def __init__(self, fusion="rgb", n_clases=40):
        super().__init__()
        self.fusion = fusion
        self.evit = S.cargar_efficientvit("b1", "ade20k")
        capa = self.evit.head.output_ops[0].op_list[-1]                # clasificador 64 -> 150 de ADE20K
        assert capa.conv.out_channels == 150
        capa.conv = nn.Conv2d(capa.conv.in_channels, n_clases, 1, bias=True)
        self.evit.backbone.requires_grad_(False)
        c_rgb = {"stage2": 64, "stage3": 128, "stage4": 256}
        if fusion == "rgb":
            self.defm = None
        else:
            self.defm = cargar_defm().requires_grad_(False)
            if fusion == "concat":
                self.fus = nn.ModuleDict({c: nn.Conv2d(128, c_rgb[c], 1, bias=False) for c, _ in PARES})
                for m in self.fus.values():
                    nn.init.zeros_(m.weight)
            elif fusion == "mlca":
                self.fus = nn.ModuleDict({c: MLCA(c_rgb[c], 128, cabezas=4) for c, _ in PARES})
            else:
                raise ValueError(fusion)

    def train(self, modo=True):
        super().train(modo)
        self.evit.backbone.eval()                                       # congelados: BN y demás en evaluación
        if self.defm is not None:
            self.defm.eval()
        return self

    def entrenables(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, rgb_norm, prof_m=None, valida=None):
        with torch.no_grad():
            f = self.evit.backbone(rgb_norm)
            if self.defm is not None:
                fd = self.defm(entrada_defm(prof_m, valida))["dense_bifpn"]
        if self.defm is not None:
            f = dict(f)
            for c, d in PARES:
                if self.fusion == "concat":
                    f[c] = f[c] + self.fus[c](fd[d].to(f[c].dtype))
                else:
                    m = validez_a_rasgos(valida, f[c].shape[-2:])
                    f[c] = self.fus[c](f[c], fd[d].to(f[c].dtype), m)
        return self.evit.head(f)["segout"]


# --------------------------------------------------------------------------------- condiciones de la profundidad
CONDICIONES = ["completa", "cruda", "aleatoria_10", "aleatoria_30", "aleatoria_50", "regiones", "sin",
               "ruido_10cm", "ruido_20cm", "ruido_50cm"]


def huecos_regiones(h, w, s, rng):
    """Corrupción *hole* de GeomPrompt (`geomprompt/corruptions.py`): max(1, round(2+18s)) rectángulos a cero, de
    alto en [max(4, 0.02h), max(5, (0.06+0.22s)h)) y ancho análogo. Devuelve la máscara de huecos (h,w) bool."""
    m = np.zeros((h, w), bool)
    for _ in range(max(1, round(2 + 18 * s))):
        rh = rng.integers(max(4, int(0.02 * h)), max(5, int((0.06 + 0.22 * s) * h)))
        rw = rng.integers(max(4, int(0.02 * w)), max(5, int((0.06 + 0.22 * s) * w)))
        y, x = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
        m[y:y + rh, x:x + rw] = True
    return m


def condicion(nombre, prof_mm, cruda_mm, semilla):
    """Profundidad (H,W) en metros y validez para una condición de evaluación, a partir de la rellenada y la cruda
    en mm. Determinista para una semilla dada (la de la imagen)."""
    rng = np.random.default_rng(semilla)
    prof = prof_mm.astype(np.float32) / 1e3
    if nombre == "completa":
        valida = prof > 0
    elif nombre == "cruda":
        prof = cruda_mm.astype(np.float32) / 1e3
        valida = prof > 0
    elif nombre.startswith("aleatoria_"):                      # Liao et al.: proporción r de píxeles a cero
        valida = (prof > 0) & (rng.random(prof.shape) >= int(nombre.split("_")[1]) / 100)
    elif nombre == "regiones":                                 # severidad media de GeomPrompt
        valida = (prof > 0) & ~huecos_regiones(*prof.shape, 0.5, rng)
    elif nombre == "sin":
        valida = np.zeros(prof.shape, bool)
    elif nombre.startswith("ruido_"):                          # gaussiano aditivo de σ cm (D-02, provisional)
        sigma = int(nombre.split("_")[1].removesuffix("cm")) / 100
        valida = prof > 0                                      # el ruido conserva valores: la máscara no lo ve
        prof = np.clip(prof + rng.normal(0, sigma, prof.shape), 0.1, None).astype(np.float32)
    else:
        raise ValueError(nombre)
    return np.where(valida, prof, 0).astype(np.float32), valida
