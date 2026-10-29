# -*- coding: utf-8 -*-
"""Segmentador RGB-D de la propuesta: EfficientViT-B1 (color) y DeFM ResNet-18 (profundidad), congelados.

Se entrenan solo la fusión y la cabeza de EfficientViT (preentrenada en ADE20K, con el clasificador cambiado a
40 clases). La fusión actúa en las tres escalas que lee la cabeza (1/8, 1/16, 1/32), emparejadas con P3-P5 de
DeFM (`09_mlca_bloque` §2.2):

- `rgb` (E1): sin profundidad.
- `concat` (E2): f_color + W·f_prof, con W (1×1, sin sesgo) iniciada en cero. Equivale a concatenar y proyectar
  con la parte de color fija en la identidad, así que E1 es el caso W = 0. Sin máscara: los huecos entran.
- `mlca` (E4): f_color + MLCA(f_color, f_prof, validez) (`mlca.py`).
- `respaldo` (E8): dos expertos, la cabeza de E1 sobre el color y la de E2 sobre la fusión, mezclados por píxel
  con una compuerta que solo mira la validez de la profundidad (`Compuerta`). Donde no hay ninguna medida
  cerca, la compuerta vale 0 y la salida es exactamente la del experto de color (D-03).

Opciones que no dependen de la fusión (D-03): `entrenable` deja ajustar etapas del codificador de color (E1u,
E7), y `aux_prof` añade una cabeza que predice la profundidad solo para entrenar (E7).

También las condiciones de evaluación de la profundidad (propuesta §7; D-01). Uso de IA: escrito con asistencia
de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import copy
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


def nueva_cabeza(cabeza, n):
    """Copia de la cabeza de EfficientViT-Seg (ADE20K) con el clasificador 64 -> 150 cambiado a n salidas."""
    c = copy.deepcopy(cabeza)
    capa = c.output_ops[0].op_list[-1]
    assert capa.conv.out_channels == 150
    capa.conv = nn.Conv2d(capa.conv.in_channels, n, 1, bias=True)
    return c


def salida(cabeza, f):
    """La cabeza de EfficientViT escribe «segout» en el diccionario que recibe: se le pasa una copia."""
    return cabeza(dict(f))["segout"]


class Compuerta(nn.Module):
    """α ∈ [0, 1] por píxel (a la resolución de la cabeza, 1/8) a partir de la validez de la profundidad:

        α = σ(w · [m_1, m_5, m_15] + b) · m_15

    m_k es la fracción de píxeles con medida en una ventana de k×k celdas (8, 40 y 120 píxeles de imagen). Las
    ventanas grandes avisan de que los rasgos de DeFM ya vienen alterados por un hueco cercano. Sin ninguna
    medida en la ventana grande, α = 0 exacto. Al inicio w = b = 0, así que α = m_15 / 2 (D-03)."""
    VENTANAS = (1, 5, 15)

    def __init__(self):
        super().__init__()
        self.lin = nn.Conv2d(len(self.VENTANAS), 1, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, valida, tam):
        m = F.adaptive_avg_pool2d(valida.float(), tam)
        ms = [m if k == 1 else F.avg_pool2d(m, k, 1, k // 2, count_include_pad=False) for k in self.VENTANAS]
        return torch.sigmoid(self.lin(torch.cat(ms, 1))) * ms[-1]


class RGBD(nn.Module):
    def __init__(self, fusion="rgb", n_clases=40, entrenable=(), aux_prof=False):
        super().__init__()
        self.fusion, self.entrenable = fusion, tuple(entrenable)
        self.evit = S.cargar_efficientvit("b1", "ade20k")
        cabeza_ade = self.evit.head                                      # clasificador 64 -> 150 de ADE20K
        self.evit.head = nueva_cabeza(cabeza_ade, n_clases)
        self.evit.backbone.requires_grad_(False)
        for e in self.entrenable:                                       # E1u, E7: «stage4» = stages[3]
            self.evit.backbone.stages[int(e.removeprefix("stage")) - 1].requires_grad_(True)
        self.cabeza_prof = None
        if aux_prof:                    # E7: arranca prediciendo 0 (la pérdida es la varianza del log de la prof.)
            self.cabeza_prof = nueva_cabeza(cabeza_ade, 1)
            capa = self.cabeza_prof.output_ops[0].op_list[-1].conv
            nn.init.zeros_(capa.weight)
            nn.init.zeros_(capa.bias)
        c_rgb = {"stage2": 64, "stage3": 128, "stage4": 256}
        if fusion == "rgb":
            self.defm = None
        else:
            self.defm = cargar_defm().requires_grad_(False)
            if fusion in ("concat", "respaldo"):
                self.fus = nn.ModuleDict({c: nn.Conv2d(128, c_rgb[c], 1, bias=False) for c, _ in PARES})
                for m in self.fus.values():
                    nn.init.zeros_(m.weight)
                if fusion == "respaldo":                                # experto de color y compuerta
                    self.cabeza_rgb = nueva_cabeza(cabeza_ade, n_clases)
                    self.compuerta = Compuerta()
            elif fusion == "mlca":
                self.fus = nn.ModuleDict({c: MLCA(c_rgb[c], 128, cabezas=4) for c, _ in PARES})
            else:
                raise ValueError(fusion)

    def train(self, modo=True):
        super().train(modo)
        self.evit.backbone.eval()                # congelado o no, BN con sus estadísticas de ADE20K
        if self.defm is not None:
            self.defm.eval()
        return self

    def entrenables(self):
        return [p for p in self.parameters() if p.requires_grad]

    def color(self, rgb_norm):
        """Rasgos de EfficientViT por etapas, como `backbone.forward`; solo las de `entrenable` llevan gradiente."""
        bb = self.evit.backbone
        with torch.no_grad():
            f = {"input": rgb_norm, "stage0": bb.input_stem(rgb_norm)}
        x = f["stage0"]
        for i, etapa in enumerate(bb.stages, 1):
            with torch.set_grad_enabled(torch.is_grad_enabled() and f"stage{i}" in self.entrenable):
                f[f"stage{i}"] = x = etapa(x)
        f["stage_final"] = x
        return f

    def forward(self, rgb_norm, prof_m=None, valida=None, todo=False):
        """Logits (B, 40, H/8, W/8). Con `todo`, un diccionario con «seg» y, según el modelo, «rgb», «fus»,
        «alfa» (E8) o «prof» (E7). En E8 «seg» es el logaritmo de la mezcla de probabilidades."""
        f = self.color(rgb_norm)
        g = f
        if self.defm is not None:
            with torch.no_grad():
                fd = self.defm(entrada_defm(prof_m, valida))["dense_bifpn"]
            g = dict(f)
            for c, d in PARES:
                if self.fusion == "mlca":
                    m = validez_a_rasgos(valida, f[c].shape[-2:])
                    g[c] = self.fus[c](f[c], fd[d].to(f[c].dtype), m)
                else:
                    g[c] = f[c] + self.fus[c](fd[d].to(f[c].dtype))
        lg = salida(self.evit.head, g)
        sal = {}
        if self.fusion == "respaldo":
            sal["rgb"], sal["fus"] = salida(self.cabeza_rgb, f), lg
            sal["alfa"] = alfa = self.compuerta(valida, lg.shape[-2:])
            p = (1 - alfa) * sal["rgb"].float().softmax(1) + alfa * lg.float().softmax(1)
            lg = p.clamp_min(1e-12).log()
        sal["seg"] = lg
        if self.cabeza_prof is not None:
            sal["prof"] = salida(self.cabeza_prof, f)
        return sal if todo else lg


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


def huecos_entrenamiento(prof, gen, p_sin=0.15, p_regiones=0.35):
    """Corrupción de la profundidad al entrenar (E8, D-03), por muestra: sin profundidad con prob. `p_sin`,
    rectángulos de `huecos_regiones` con severidad U(0, 1) con prob. `p_regiones`, completa en el resto.
    prof (B,1,H,W) en mm; devuelve una copia con los huecos a 0."""
    sal = prof.clone()
    for i in range(len(prof)):
        u = torch.rand(1, generator=gen).item()
        if u < p_sin:
            sal[i] = 0
        elif u < p_sin + p_regiones:
            rng = np.random.default_rng(int(torch.randint(2**31 - 1, (1,), generator=gen)))
            m = huecos_regiones(*prof.shape[-2:], float(rng.random()), rng)
            sal[i][:, torch.from_numpy(m).to(prof.device)] = 0
    return sal


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
