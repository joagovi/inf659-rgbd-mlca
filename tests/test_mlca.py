# -*- coding: utf-8 -*-
"""El bloque MLCA y la máscara, sin redes preentrenadas (D-06)."""
import pytest
import torch

from mlca import MLCA, validez_a_rasgos
from rgbd import preparar_mascara


def bloque(semilla=0, c_rgb=64, c_prof=128, salida_aleatoria=True):
    torch.manual_seed(semilla)
    b = MLCA(c_rgb, c_prof, cabezas=4)
    if salida_aleatoria:                       # como después de entrenar: la salida ya no es cero
        torch.nn.init.normal_(b.salida.weight, std=0.1)
    return b


def entradas(semilla=1, B=2, h=6, w=8):
    g = torch.Generator().manual_seed(semilla)
    return torch.randn(B, 64, h, w, generator=g), torch.randn(B, 128, h, w, generator=g)


def test_sin_profundidad_aporta_cero_exacto():
    b, (fr, fp) = bloque(), entradas()
    assert torch.equal(b(fr, fp, torch.zeros(2, 1, 6, 8)), fr)


def test_celdas_invalidas_no_influyen_aunque_sean_nan():
    b, (fr, fp) = bloque(), entradas()
    m = torch.ones(2, 1, 6, 8)
    m[:, :, :3] = 0
    fp2 = fp.clone()
    fp2[:, :, :3] = float("nan")
    a, c = b(fr, fp, m), b(fr, fp2, m)
    assert torch.isfinite(c).all() and torch.allclose(a, c, atol=0, rtol=0)


@pytest.mark.parametrize("modo", ["fraccional", "binaria", "ninguna"])
def test_con_profundidad_completa_los_tres_modos_coinciden(modo):
    """E3 y E4 son la misma red: con m = 1 en todas partes calculan exactamente lo mismo."""
    b, (fr, fp) = bloque(), entradas()
    unos = torch.ones(2, 1, 6, 8)
    assert torch.equal(b(fr, fp, preparar_mascara(unos, modo)), b(fr, fp, unos))


def test_binaria_y_fraccional_difieren_con_huecos():
    b, (fr, fp) = bloque(), entradas()
    m = torch.rand(2, 1, 6, 8, generator=torch.Generator().manual_seed(3))
    assert not torch.allclose(b(fr, fp, preparar_mascara(m, "binaria")), b(fr, fp, preparar_mascara(m, "fraccional")))
    mb = preparar_mascara(m, "binaria", 0.5)
    assert set(mb.unique().tolist()) <= {0.0, 1.0} and torch.equal(mb, (m > 0.5).float())


def test_validez_por_areas():
    v = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    v[..., :8, :] = True
    v[..., 8:12, :4] = True
    m = validez_a_rasgos(v, (2, 2))
    assert torch.allclose(m, torch.tensor([[[[1.0, 1.0], [0.25, 0.0]]]]))


def test_claves_y_valores_constantes_suman_un_sesgo():
    """Con rasgos de profundidad constantes (P3, P5 de DeFM), MLCA suma lo mismo en todas las posiciones, para
    cualquier color y cualquier máscara con alguna celda válida; sin ninguna, nada. Es el colapso de E4.

    Por cabeza, o_i = v · φ(q_i)ᵀz / (φ(q_i)ᵀz + ε): el valor v (común a todas las celdas) salvo el factor de ε, o
    0 si la ReLU anula toda la consulta de esa cabeza. Con la salida igual a la identidad se ve o directamente."""
    b = bloque()
    with torch.no_grad():
        b.salida.weight.copy_(torch.eye(64).view(64, 64, 1, 1))
    fr1, _ = entradas(1)
    fr2, _ = entradas(2)
    fp = torch.randn(1, 128, 1, 1).expand(2, 128, 6, 8).contiguous()
    m1 = torch.ones(2, 1, 6, 8)
    m2 = (torch.rand(2, 1, 6, 8, generator=torch.Generator().manual_seed(4)) > 0.7).float()
    m2[:, :, 0, 0] = 1
    v = b._tokens(b.v(fp))[0, :, 0]                                   # (cabezas, d): el valor de cada cabeza
    for fr in (fr1, fr2):
        for m in (m1, m2):
            o = (b(fr, fp, m) - fr).reshape(2, 4, 16, 48).permute(0, 3, 1, 2)   # (B, N, cabezas, d)
            es_v = (o - v).abs().amax(-1) < 1e-4 * (1 + v.abs().amax(-1))
            es_0 = o.abs().amax(-1) == 0
            assert (es_v | es_0).all()
            assert es_v.float().mean() > 0.8
    assert torch.equal(b(fr1, fp, torch.zeros_like(m1)), fr1)


def test_gradientes_al_inicio_y_tras_un_paso():
    """Con la salida en cero solo ella recibe gradiente; después de un paso, también Q, K y V."""
    b = bloque(salida_aleatoria=False)
    fr, fp = entradas()
    fr.requires_grad_(True)
    m = torch.ones(2, 1, 6, 8)
    b(fr, fp, m).pow(2).sum().backward()
    assert b.salida.weight.grad.abs().sum() > 0
    for p in (b.q.weight, b.k.weight, b.v.weight):
        assert p.grad is None or p.grad.abs().sum() == 0
    with torch.no_grad():
        b.salida.weight -= 0.1 * b.salida.weight.grad
    b.zero_grad()
    b(fr, fp, m).pow(2).sum().backward()
    for p in (b.q.weight, b.k.weight, b.v.weight, b.salida.weight):
        assert p.grad.abs().sum() > 0


def test_estable_con_valores_grandes_y_muchos_tokens_en_bf16():
    b = bloque()
    fr, fp = torch.randn(1, 64, 60, 80) * 50, torch.randn(1, 128, 60, 80) * 50
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o = b(fr, fp, torch.ones(1, 1, 60, 80))
    assert torch.isfinite(o).all()


def test_coste_lineal_en_tokens():
    """Las sumas S y z no dependen del número de consultas: duplicar N duplica (no cuadruplica) los FLOPs."""
    from torch.utils.flop_counter import FlopCounterMode
    b = bloque()

    def flops(h, w):
        fr, fp = torch.randn(1, 64, h, w), torch.randn(1, 128, h, w)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            b(fr, fp, torch.ones(1, 1, h, w))
        return fc.get_total_flops()
    r = flops(32, 64) / flops(32, 32)
    assert 1.9 < r < 2.1
