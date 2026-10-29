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
  cerca, la compuerta vale 0 y la salida es exactamente la del experto de color (D-03). Con `experto_fijo`
  (E9), el experto de color es un E1u ya entrenado, con su última etapa y su cabeza congeladas (D-04).

Qué rasgos de profundidad (D-05): en los pesos publicados de DeFM ResNet-18, P3 y P5 no dependen de la entrada
(las convoluciones de salida de la BiFPN para esos niveles tienen pesos de ~1e-9, y en P5 también los pesos de
fusión son 0): hasta E9, solo P4 (1/16) lleva información de la profundidad, y en 1/8 y 1/32 la fusión suma una
constante. Con `rasgos_prof="c3p4c5"` (E10, E11), en 1/8 y 1/32 se usan las etapas del ResNet de DeFM (C3, C5),
normalizadas por posición sin parámetros, como DeFM hace con sus salidas densas; en 1/16 sigue P4.

Dónde entra la profundidad (D-05): en todas, en los rasgos que lee la cabeza, después del codificador de color
(que nunca la ve). Con `codificador` (E11, solo `concat`), la última etapa del color lee además f_1/16 + W'·P4,
con W' propia (1×1, iniciada en cero), como la rectificación de CMX antes de la etapa siguiente; con W' = 0 es la
misma red que sin ella.

Opciones que no dependen de la fusión (D-03): `entrenable` deja ajustar etapas del codificador de color (E1u,
E7, E2u, E9, E10, E11), y `aux_prof` añade una cabeza que predice la profundidad solo para entrenar (E7).

También las condiciones de evaluación de la profundidad (propuesta §7; D-01). Uso de IA: escrito con asistencia
de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import efficientvit_seg as S
from mlca import MLCA, validez_a_rasgos

DEFM_MEAN = torch.tensor([0.248880, 0.495620, 0.492858]).view(1, 3, 1, 1)   # defm/utils/utils.py
DEFM_STD = torch.tensor([0.139357, 0.271314, 0.297177]).view(1, 3, 1, 1)
PARES = [("stage2", "P3"), ("stage3", "P4"), ("stage4", "P5")]           # color y profundidad a la misma escala
RASGOS_PROF = {"p3p4p5": PARES, "c3p4c5": [("stage2", "C3"), ("stage3", "P4"), ("stage4", "C5")]}   # D-05
CANALES_PROF = {"P3": 128, "P4": 128, "P5": 128, "C3": 128, "C5": 512}


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


def rasgos_defm(defm, prof_m, valida, cuales="p3p4p5"):
    """Rasgos densos de DeFM para la fusión. «p3p4p5»: las salidas de su BiFPN (`dense_bifpn`), como hasta D-04.
    «c3p4c5» (D-05): C3 y C5 del ResNet, con LayerNorm por posición sin parámetros, y P4 de la BiFPN, calculados
    en una sola pasada como en `ResNetBiFPN.forward` del código oficial."""
    x = entrada_defm(prof_m, valida)
    if cuales == "p3p4p5":
        return defm(x)["dense_bifpn"]
    c = OrderedDict(defm.backbone(x).items())
    por_posicion = lambda t, ln: ln(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # noqa: E731
    sin_param = lambda t: F.layer_norm(t, t.shape[-1:])                          # noqa: E731
    return {"C3": por_posicion(c["c3"], sin_param), "P4": por_posicion(defm.fpn(c)["c4"], defm.norm_patch),
            "C5": por_posicion(c["c5"], sin_param)}


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
    def __init__(self, fusion="rgb", n_clases=40, entrenable=(), aux_prof=False, experto_fijo=False,
                 rasgos_prof="p3p4p5", codificador=False):
        super().__init__()
        self.fusion, self.entrenable, self.codificador = fusion, tuple(entrenable), codificador
        self.rasgos_prof, self.pares = rasgos_prof, RASGOS_PROF[rasgos_prof]
        self.etapa4_rgb = None
        if codificador and fusion != "concat":
            raise ValueError("codificador solo con la fusión «concat»")
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
                self.fus = nn.ModuleDict({c: nn.Conv2d(CANALES_PROF[d], c_rgb[c], 1, bias=False)
                                          for c, d in self.pares})
                for m in self.fus.values():
                    nn.init.zeros_(m.weight)
                if codificador:             # E11: W' de la profundidad de 1/16 a la entrada de la última etapa
                    self.fus_cod = nn.Conv2d(CANALES_PROF["P4"], c_rgb["stage3"], 1, bias=False)
                    nn.init.zeros_(self.fus_cod.weight)
                if fusion == "respaldo":                                # experto de color y compuerta
                    self.cabeza_rgb = nueva_cabeza(cabeza_ade, n_clases)
                    self.compuerta = Compuerta()
                    if experto_fijo:        # E9: copia propia de la última etapa; los pesos, de `fijar_experto`
                        self.etapa4_rgb = copy.deepcopy(self.evit.backbone.stages[3]).requires_grad_(False)
                        self.cabeza_rgb.requires_grad_(False)
            elif fusion == "mlca":
                self.fus = nn.ModuleDict({c: MLCA(c_rgb[c], CANALES_PROF[d], cabezas=4) for c, d in self.pares})
            else:
                raise ValueError(fusion)
        if experto_fijo and self.etapa4_rgb is None:
            raise ValueError("experto_fijo solo con la fusión «respaldo»")

    def train(self, modo=True):
        super().train(modo)
        self.evit.backbone.eval()                # congelado o no, BN con sus estadísticas de ADE20K
        if self.defm is not None:
            self.defm.eval()
        if self.etapa4_rgb is not None:          # E9: el experto de color no cambia, tampoco sus BN
            self.etapa4_rgb.eval()
            self.cabeza_rgb.eval()
        return self

    def entrenables(self):
        return [p for p in self.parameters() if p.requires_grad]

    def fijar_experto(self, estado):
        """E9: carga como experto de color la última etapa y la cabeza de un E1u ya entrenado (su cabeza.pt). Las
        estadísticas de BN de esa etapa no se guardan en cabeza.pt; son las de ADE20K, las de la copia."""
        nuevo = {}
        for k, v in estado.items():
            if k.startswith("evit.backbone.stages.3."):
                nuevo["etapa4_rgb." + k.removeprefix("evit.backbone.stages.3.")] = v
            elif k.startswith("evit.head."):
                nuevo["cabeza_rgb." + k.removeprefix("evit.head.")] = v
            else:
                raise KeyError(f"{k}: no es de un E1u")
        info = self.load_state_dict(nuevo, strict=False)
        faltan = [k for k in info.missing_keys if k.startswith(("etapa4_rgb.", "cabeza_rgb."))]
        assert not info.unexpected_keys
        assert all(k.startswith("etapa4_rgb.") and k.endswith(("running_mean", "running_var", "num_batches_tracked"))
                   for k in faltan), faltan

    def etapa(self, i, x):
        """Etapa i (1-4) del codificador de color; solo las de `entrenable` llevan gradiente."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and f"stage{i}" in self.entrenable):
            return self.evit.backbone.stages[i - 1](x)

    def color(self, rgb_norm, etapas=4):
        """Rasgos de EfficientViT por etapas, como `backbone.forward`, hasta la etapa `etapas`."""
        bb = self.evit.backbone
        with torch.no_grad():
            f = {"input": rgb_norm, "stage0": bb.input_stem(rgb_norm)}
        x = f["stage0"]
        for i in range(1, etapas + 1):
            f[f"stage{i}"] = x = self.etapa(i, x)
        f["stage_final"] = x
        return f

    def forward(self, rgb_norm, prof_m=None, valida=None, todo=False):
        """Logits (B, 40, H/8, W/8). Con `todo`, un diccionario con «seg» y, según el modelo, «rgb», «fus»,
        «alfa» (E8) o «prof» (E7). En E8 «seg» es el logaritmo de la mezcla de probabilidades."""
        f = self.color(rgb_norm, 3 if self.codificador else 4)
        g = f
        if self.defm is not None:
            with torch.no_grad():
                fd = rasgos_defm(self.defm, prof_m, valida, self.rasgos_prof)
            if self.codificador:            # E11: la última etapa del color también lee la profundidad
                x = f["stage3"] + self.fus_cod(fd["P4"].to(f["stage3"].dtype))
                f["stage4"] = f["stage_final"] = self.etapa(4, x)
            g = dict(f)
            for c, d in self.pares:
                if self.fusion == "mlca":
                    m = validez_a_rasgos(valida, f[c].shape[-2:])
                    g[c] = self.fus[c](f[c], fd[d].to(f[c].dtype), m)
                else:
                    g[c] = f[c] + self.fus[c](fd[d].to(f[c].dtype))
        lg = salida(self.evit.head, g)
        sal = {}
        if self.fusion == "respaldo":
            fr = f
            if self.etapa4_rgb is not None:                         # E9: la última etapa de E1u
                with torch.no_grad():
                    x = self.etapa4_rgb(f["stage3"])
                fr = {**f, "stage4": x, "stage_final": x}
            sal["rgb"], sal["fus"] = salida(self.cabeza_rgb, fr), lg
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
