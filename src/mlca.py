# -*- coding: utf-8 -*-
"""Atención cruzada lineal enmascarada (MLCA): el color pregunta, la profundidad válida responde.

    o_i = φ(q_i)ᵀ S / (φ(q_i)ᵀ z + ε),   S = Σ_j m_j φ(k_j) v_jᵀ,   z = Σ_j m_j φ(k_j),   φ = ReLU

q sale de los rasgos de color; k y v, de los de profundidad; m_j ∈ [0, 1] es la fracción de píxeles con
profundidad medida en la celda j. Una celda con m_j = 0 no entra ni en el resumen S ni en el normalizador z,
así que su contenido (aunque sea NaN) no influye. La proyección de salida no tiene sesgo y empieza en cero:
sin profundidad (m = 0 en todas partes) S = 0, z = 0, o = 0 y el bloque devuelve exactamente los rasgos de
color, en cualquier momento del entrenamiento, no solo al inicio.

Lo usa `09_mlca_bloque.ipynb`. Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver
`DECISIONES.md`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLCA(nn.Module):
    """Bloque MLCA residual: (rasgos de color, rasgos de profundidad, validez) -> rasgos de color."""

    def __init__(self, c_rgb, c_prof, dim=None, cabezas=4, eps=1e-6):
        super().__init__()
        dim = dim or c_rgb
        assert dim % cabezas == 0, (dim, cabezas)
        self.cabezas, self.eps = cabezas, eps
        self.q = nn.Conv2d(c_rgb, dim, 1)
        self.k = nn.Conv2d(c_prof, dim, 1)
        self.v = nn.Conv2d(c_prof, dim, 1)
        self.salida = nn.Conv2d(dim, c_rgb, 1, bias=False)
        nn.init.zeros_(self.salida.weight)

    def _tokens(self, x):
        """(B, C, H, W) -> (B, cabezas, N, C/cabezas)."""
        b, c, h, w = x.shape
        return x.reshape(b, self.cabezas, c // self.cabezas, h * w).transpose(-1, -2)

    def forward(self, f_rgb, f_prof, m):
        """f_rgb (B,Cc,H,W), f_prof (B,Cp,H,W), m (B,1,H,W) en [0, 1] a la resolución de los rasgos."""
        b, _, h, w = f_rgb.shape
        q = F.relu(self._tokens(self.q(f_rgb)))
        k = F.relu(self._tokens(self.k(f_prof)))
        v = self._tokens(self.v(f_prof))
        mj = m.reshape(b, 1, h * w, 1)
        # torch.where y no k * m: 0 * NaN = NaN, y un hueco con basura no debe contaminar el resumen
        k = torch.where(mj > 0, k * mj, torch.zeros_like(k))
        v = torch.where(mj > 0, v, torch.zeros_like(v))
        # Las sumas sobre N, en FP32: con ReLU y miles de tokens desbordan en FP16 (EfficientViT en TensorRT)
        with torch.autocast(device_type=f_rgb.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            S = k.transpose(-1, -2) @ v                                   # (B, cabezas, d, d)
            z = k.sum(dim=-2)                                             # (B, cabezas, d)
            o = (q @ S) / ((q @ z.unsqueeze(-1)) + self.eps)              # (B, cabezas, N, d)
        o = o.to(f_rgb.dtype).transpose(-1, -2).reshape(b, -1, h, w)
        return f_rgb + self.salida(o)


def validez_a_rasgos(valida, tam):
    """Máscara de validez a resolución de imagen (B,1,H,W) -> fracción válida por celda de rasgos (B,1,h,w).

    Con un promedio por áreas, una celda sin ningún píxel medido queda en 0 exacto y sale del resumen.
    """
    return F.adaptive_avg_pool2d(valida.float(), tam)


def atencion_softmax_cruzada(f_rgb, f_prof, cabezas=4):
    """Referencia de coste cuadrático: atención cruzada softmax sin proyecciones ni máscara."""
    b, c, h, w = f_rgb.shape
    t = lambda x: x.reshape(b, cabezas, c // cabezas, h * w).transpose(-1, -2).contiguous()
    return F.scaled_dot_product_attention(t(f_rgb), t(f_prof), t(f_prof))
