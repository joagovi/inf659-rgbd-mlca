# -*- coding: utf-8 -*-
"""Solo con CUDA: la red en bf16, las sumas de MLCA en fp32 y la memoria (D-06). Se saltan sin GPU."""
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.modelo]


def test_e4c_en_bf16_finito_y_ruta_de_color_exacta():
    import entrenar_rgbd as E
    torch.manual_seed(0)
    m = E.construir("E4c").to("cuda").eval()
    x = torch.randn(2, 3, 480, 640, device="cuda")
    p = 1 + 4 * torch.rand(2, 1, 480, 640, device="cuda")
    v = torch.ones_like(p, dtype=torch.bool)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.isfinite(m(x, p, v)).all()
        s = m(x, torch.zeros_like(p), torch.zeros_like(v), rasgos=True)
    assert all(torch.equal(s["f_color"][c], s["f_fusion"][c]) for c in s["f_color"])


def test_memoria_pico_razonable():
    import entrenar_rgbd as E
    torch.cuda.reset_peak_memory_stats()
    m = E.construir("E4cd").to("cuda").train()
    x = torch.randn(8, 3, 480, 640, device="cuda")
    p = 1 + 4 * torch.rand(8, 1, 480, 640, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x, p, p > 0).float().mean().backward()
    assert torch.cuda.max_memory_allocated() < 8 * 2**30     # cabe en la RTX 4060 Laptop de 8 GB
