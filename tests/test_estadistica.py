# -*- coding: utf-8 -*-
"""Las tablas se recalculan desde los resultado.json de la v3 y reproducen lo publicado en DECISIONES.md (D-06)."""
from pathlib import Path

import numpy as np
import pytest

import estadistica as S

V3 = Path(__file__).resolve().parents[1] / "runs" / "rgbd"
pytestmark = pytest.mark.skipif(not (V3 / "E1_lr0.001_s0").exists(), reason="sin los resultados de la v3")

PUBLICADO = {  # (exp, condición): (media, desviación) de DECISIONES.md / LEEME_EQUIPO.md
    ("E1", "completa"): (44.99, 0.30), ("E2", "completa"): (47.31, 0.26), ("E2", "sin"): (42.25, 0.45),
    ("E4", "completa"): (45.24, 0.29), ("E4", "sin"): (0.78, 1.22), ("E4", "aleatoria_50"): (41.89, 1.40),
    ("E5", "completa"): (45.48, 0.21), ("E5", "sin"): (45.74, 0.10), ("E1u", "completa"): (47.31, 0.10),
    ("E2u", "completa"): (48.59, 0.15), ("E2u", "sin"): (36.84, 0.70), ("E9", "completa"): (47.76, 0.09),
    ("E8", "completa"): (48.17, 0.33), ("E2h", "sin"): (44.59, 0.20)}


@pytest.fixture(scope="module")
def tablas():
    return S.tablas(V3)[1]


@pytest.mark.parametrize("clave", list(PUBLICADO))
def test_reproduce_lo_publicado(tablas, clave):
    e, c = clave
    r = tablas["experimentos"][e]["condiciones"][c]
    assert r["n"] == 3
    assert round(r["media"], 2) == PUBLICADO[clave][0] and round(r["de"], 2) == PUBLICADO[clave][1]


def test_intervalo_t_con_tres_semillas(tablas):
    r = tablas["experimentos"]["E2"]["condiciones"]["completa"]
    assert abs(r["ic95"] - 4.303 * r["de"] / np.sqrt(3)) < 1e-12


D06 = {  # los resultados de D-06 en DECISIONES.md, si ya se corrió la cola (2026-11-01)
    ("E2c", "completa"): (47.53, 0.17), ("E4c", "cruda"): (40.56, 0.34), ("E3cd", "sin"): (45.31, 0.25),
    ("E4cd", "sin"): (46.16, 0.26)}


def test_factorial_pendiente_no_se_inventa():
    """Sin las corridas de D-06, el factorial queda pendiente; con ellas, sale de los resultados y reproduce lo
    publicado en DECISIONES.md (D-06)."""
    md, t = S.tablas(V3)
    if not any(V3.glob("E4cd_lr*_s0")):
        assert "Pendiente: faltan E3c, E4c, E3cd, E4cd" in md
        return
    assert "Pendiente: faltan E3c, E4c, E3cd, E4cd" not in md
    assert "| cruda | -0.63 ± 0.23 | +0.01 ± 0.17 | +3.71 ± 0.30 | +0.64 ± 0.05 |" in md
    for (e, c), (media, de) in D06.items():
        r = t["experimentos"][e]["condiciones"][c]
        assert r["n"] == 3 and round(r["media"], 2) == media and round(r["de"], 2) == de, (e, c)


def test_bootstrap_pareado(tmp_path):
    rng = np.random.default_rng(0)
    carpetas = []
    for nombre, extra in (("a", 5), ("b", 0)):
        d = tmp_path / nombre
        d.mkdir()
        u = rng.integers(50, 100, (30, 40))
        i = np.minimum(u, rng.integers(10, 40, (30, 40)) + extra)
        np.savez(d / "por_imagen.npz", completa_inter=i, completa_union=u)
        carpetas.append(d)
    r = S.bootstrap_pareado([carpetas[0]], [carpetas[1]], "completa", B=200)
    assert r["dif"] > 0 and r["ic95"][0] < r["dif"] < r["ic95"][1]
