# -*- coding: utf-8 -*-
"""Condiciones de la profundidad, codificación de DeFM y datos (D-06)."""
import numpy as np
import pytest
import torch

import nyuv2
import rgbd

H, W = 96, 128


def prof_mm(semilla=0):
    rng = np.random.default_rng(semilla)
    p = (1000 + 3000 * rng.random((H, W))).astype(np.uint16)
    c = p.copy()
    c[:, :8] = 0
    return p, c


@pytest.mark.parametrize("cond", rgbd.CONDICIONES + rgbd.CONDICIONES_EXTRA)
def test_deterministas_y_en_metros(cond):
    p, c = prof_mm()
    a, va = rgbd.condicion(cond, p, c, semilla=7)
    b, vb = rgbd.condicion(cond, p, c, semilla=7)
    assert np.array_equal(a, b) and np.array_equal(va, vb)
    assert a.dtype == np.float32 and va.dtype == bool
    assert (a[~va] == 0).all() and (a[va] > 0).all()
    assert a.max() <= 4.0 + 6 * 0.5                         # mm -> m (más el ruido de 50 cm)
    if cond not in ("sin",) and not cond.startswith("ruido"):
        assert a[va].max() <= p.max() / 1e3 + 1e-6


def test_semillas_distintas_dan_huecos_distintos():
    p, c = prof_mm()
    _, v1 = rgbd.condicion("aleatoria_30", p, c, semilla=1)
    _, v2 = rgbd.condicion("aleatoria_30", p, c, semilla=2)
    assert not np.array_equal(v1, v2)


@pytest.mark.parametrize("r", [10, 30, 50])
def test_aleatoria_quita_la_fraccion_nominal(r):
    p, c = prof_mm()
    fr = [1 - rgbd.condicion(f"aleatoria_{r}", p, c, semilla=s)[1].mean() for s in range(20)]
    assert abs(np.mean(fr) - r / 100) < 0.01


def test_sin_ruido_cruda_y_regiones():
    p, c = prof_mm()
    assert not rgbd.condicion("sin", p, c, 0)[1].any()
    d, v = rgbd.condicion("ruido_50cm", p, c, 0)
    assert v.all() and d.min() >= 0.1 and not np.allclose(d, p / 1e3)   # la máscara no ve el ruido
    d, v = rgbd.condicion("cruda", p, c, 0)
    assert np.array_equal(v, c > 0)
    fr = [1 - rgbd.condicion("regiones", p, c, s)[1].mean() for s in range(10)]
    fr1 = [1 - rgbd.condicion("regiones_100", p, c, s)[1].mean() for s in range(10)]
    assert 0 < np.mean(fr) < np.mean(fr1) < 1


def test_entrada_defm_huecos_y_minmax_valido():
    d = torch.tensor(np.linspace(0.5, 5, H * W, dtype=np.float32).reshape(1, 1, H, W))
    v = torch.ones_like(d, dtype=torch.bool)
    v[..., :10] = False
    d = torch.where(v, d, torch.zeros_like(d))
    x = rgbd.entrada_defm(d, v)
    hueco = (-rgbd.DEFM_MEAN / rgbd.DEFM_STD).view(3)
    assert torch.allclose(x[0, :, 0, 0], hueco)                # huecos = 0 antes de normalizar, como el oficial
    c3 = x[0, 2] * rgbd.DEFM_STD[0, 2, 0, 0] + rgbd.DEFM_MEAN[0, 2, 0, 0]
    assert abs(float(c3[v[0, 0]].min())) < 1e-6 and abs(float(c3[v[0, 0]].max()) - 1) < 1e-6   # D-01


def test_entrada_defm_igual_a_la_oficial_sin_huecos():
    """Sin huecos, la codificación de D-01 es la de `convert_to_3channel_metric`."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verificar_defm import oficial_3canales
    d = np.linspace(0.5, 9.5, H * W, dtype=np.float32).reshape(H, W)
    a = oficial_3canales(d)
    b = rgbd.entrada_defm(torch.from_numpy(d)[None, None], torch.from_numpy(d > 0)[None, None])
    assert torch.allclose(a, b, atol=1e-5)
    z = np.zeros_like(d)
    assert torch.allclose(oficial_3canales(z), rgbd.entrada_defm(torch.from_numpy(z)[None, None],
                                                                  torch.zeros(1, 1, H, W, dtype=torch.bool)))


def test_datos_sinteticos_como_la_cache():
    d = nyuv2.datos_sinteticos("entrenamiento")
    assert d["rgb"].dtype == np.uint8 and d["prof"].dtype == np.uint16 and d["etiqueta"].max() <= 40
    assert (d["prof"] > 0).all()                             # rellenada sin ceros, como `depths`
    assert ((d["cruda"] == 0) | (d["cruda"] == d["prof"])).all() and (d["cruda"] == 0).any()
    tr, va = nyuv2.validacion_por_escenas()
    assert len(set(tr) & set(va)) == 0
