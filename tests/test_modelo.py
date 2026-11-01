# -*- coding: utf-8 -*-
"""La red completa con la arquitectura real (EfficientViT-B1 + DeFM ResNet-18) y pesos aleatorios (D-06).

Los pesos no cambian lo que se comprueba aquí: qué ve cada experimento, qué se entrena, que E3 y E4 sean la
misma red, que sin profundidad E4 sea la ruta de color, y que la extracción de rasgos sea la oficial."""
import copy

import pytest
import torch
import torch.nn.functional as F

import entrenar_rgbd as E
import rgbd

pytestmark = pytest.mark.modelo
H, W = 96, 128


def construir(exp, semilla=0):
    torch.manual_seed(semilla)
    return E.construir(exp).eval()


@pytest.fixture(scope="module")
def entrada():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, H, W, generator=g)
    p = 1 + 4 * torch.rand(2, 1, H, W, generator=g)
    return x, p, torch.ones_like(p, dtype=torch.bool)


def aleatorizar_fusion(m, semilla=1):
    g = torch.Generator().manual_seed(semilla)
    with torch.no_grad():
        for p in m.fus.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return m


@torch.no_grad()
def test_e1_ignora_la_profundidad(entrada):
    x, p, v = entrada
    m = construir("E1")
    assert m.defm is None
    ref = m(x)
    assert torch.equal(ref, m(x, p, v)) and torch.equal(ref, m(x, torch.zeros_like(p), torch.zeros_like(v)))


@torch.no_grad()
@pytest.mark.parametrize("exp", ["E2", "E4", "E2c", "E3c", "E4c", "E4cf"])
def test_al_inicio_la_fusion_es_e1(entrada, exp):
    """Fusión iniciada en cero y la misma inicialización de la cabeza con la misma semilla: logits iguales a E1."""
    x, p, v = entrada
    assert torch.equal(construir("E1")(x), construir(exp)(x, p, v))


def test_e3_y_e4_son_la_misma_red():
    a, b, c = construir("E3c"), construir("E4c"), construir("E4cf")
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert sa.keys() == sb.keys() == sc.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]) and torch.equal(sa[k], sc[k]), k
    ent = lambda m: [n for n, q in m.named_parameters() if q.requires_grad]   # noqa: E731
    assert ent(a) == ent(b) == ent(c)
    assert (a.mascara, b.mascara, c.mascara) == ("ninguna", "binaria", "fraccional")
    da, db = (E.EXPERIMENTOS[e] for e in ("E3c", "E4c"))
    import dataclasses
    assert dataclasses.replace(da, mascara="binaria", que=db.que) == db


@torch.no_grad()
@pytest.mark.parametrize("exp", ["E4", "E4c", "E4cf"])
def test_sin_profundidad_e4_es_la_ruta_de_color(entrada, exp):
    """Con cualquier peso de la fusión, sin ninguna medida la cabeza recibe exactamente los rasgos de color."""
    x, p, v = entrada
    m = aleatorizar_fusion(construir(exp))
    s = m(x, torch.zeros_like(p), torch.zeros_like(v), rasgos=True)
    for c in s["f_color"]:
        assert torch.equal(s["f_color"][c], s["f_fusion"][c])
    assert torch.equal(s["seg"], rgbd.salida(m.evit.head, m.color(x)))


@torch.no_grad()
def test_sin_profundidad_e3_no_es_la_ruta_de_color(entrada):
    x, p, v = entrada
    m = aleatorizar_fusion(construir("E3c"))
    s = m(x, torch.zeros_like(p), torch.zeros_like(v), rasgos=True)
    assert any(not torch.equal(s["f_color"][c], s["f_fusion"][c]) for c in s["f_color"])


@torch.no_grad()
def test_con_profundidad_completa_e3_igual_a_e4(entrada):
    x, p, v = entrada
    a = aleatorizar_fusion(construir("E3c"))
    b = construir("E4c")
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(x, p, v), b(x, p, v))
    v2 = v.clone()
    v2[..., : W // 2] = False
    p2 = torch.where(v2, p, torch.zeros_like(p))
    assert not torch.equal(a(x, p2, v2), b(x, p2, v2))


@torch.no_grad()
@pytest.mark.parametrize("exp", ["E11", "E4e"])
def test_al_inicio_el_codificador_no_cambia_nada(entrada, exp):
    """W' = 0 (E11) o la salida de MLCA = 0 (D-07): al inicio, la misma red que E10, que no toca el codificador."""
    x, p, v = entrada
    assert torch.equal(construir("E10")(x, p, v), construir(exp)(x, p, v))


@torch.no_grad()
@pytest.mark.parametrize("exp,ruta_de_color", [("E4e", True), ("E3e", False)])
def test_sin_profundidad_la_mlca_del_codificador(entrada, exp, ruta_de_color):
    """D-07: con cualquier peso de la MLCA del codificador y sin ninguna medida, con máscara la última etapa lee
    exactamente los rasgos de color (las sumas de la cabeza, en cero, no añaden nada); sin máscara, no."""
    x, p, v = entrada
    m = construir(exp)
    g = torch.Generator().manual_seed(1)
    for q in m.fus_cod.parameters():
        q.copy_(0.05 * torch.randn(q.shape, generator=g))
    sin = m(x, torch.zeros_like(p), torch.zeros_like(v))
    assert torch.equal(sin, rgbd.salida(m.evit.head, m.color(x))) == ruta_de_color
    assert not torch.equal(m(x, p, v), sin)


def test_e3e_y_e4e_son_la_misma_red():
    a, b = construir("E3e"), construir("E4e")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(k.startswith("fus_cod.q.") for k in sa)
    assert (a.mascara, b.mascara, a.codificador, b.codificador) == ("ninguna", "binaria", "mlca", "mlca")
    assert E.equivalentes_en_entrenamiento("E4e", "E3e")
    assert not E.equivalentes_en_entrenamiento("E4ed", "E3ed") and not E.equivalentes_en_entrenamiento("E11", "E4e")


@pytest.mark.parametrize("kw", [dict(fusion="mlca", codificador="mlca"), dict(fusion="concat", codificador="suma"),
                                dict(fusion="concat", codificador=True, mascara="binaria")])
def test_opciones_invalidas_del_codificador(kw):
    with pytest.raises(ValueError):
        rgbd.RGBD(**kw)


@pytest.mark.parametrize("exp,esperado", [
    ("E1", {"evit.head"}), ("E2", {"evit.head", "fus"}), ("E4c", {"evit.head", "fus"}),
    ("E1u", {"evit.head", "evit.backbone.stages.3"}), ("E2u", {"evit.head", "fus", "evit.backbone.stages.3"}),
    ("E4e", {"evit.head", "fus", "fus_cod", "evit.backbone.stages.3"})])
def test_solo_se_entrena_lo_previsto(entrada, exp, esperado):
    x, p, v = entrada
    m = construir(exp)
    m.train()
    if m.defm is not None:
        aleatorizar_fusion(m)
    m(x, p, v).pow(2).mean().backward()
    con_grad = {n for n, q in m.named_parameters() if q.grad is not None and q.grad.abs().sum() > 0}
    for n, q in m.named_parameters():
        prefijo = next((e for e in esperado if n.startswith(e)), None)
        assert q.requires_grad == (prefijo is not None), n
    assert all(any(n.startswith(e) for e in esperado) for n in con_grad)
    assert all(any(n.startswith(e) for n in con_grad) for e in esperado)
    assert not m.evit.backbone.training and (m.defm is None or not m.defm.training)   # BN de ADE20K, GN de DeFM


@torch.no_grad()
@pytest.mark.parametrize("rasgos", ["p3p4p5", "c3p4c5"])
def test_correspondencia_espacial_y_de_canales(entrada, rasgos):
    x, p, v = entrada
    m = construir("E2" if rasgos == "p3p4p5" else "E2c")
    f = m.color(x)
    fd = rgbd.rasgos_defm(m.defm, p, v, rasgos)
    for c, d in m.pares:
        assert f[c].shape[-2:] == fd[d].shape[-2:], (c, d)
        assert m.fus[c].in_channels == fd[d].shape[1] == rgbd.CANALES_PROF[d]


@torch.no_grad()
def test_extraccion_de_rasgos_es_la_oficial(entrada):
    _, p, v = entrada
    m = construir("E2")
    x = rgbd.entrada_defm(p, v)
    o = m.defm(x)["dense_bifpn"]
    a = rgbd.rasgos_defm(m.defm, p, v, "p3p4p5")
    b = rgbd.rasgos_defm(m.defm, p, v, "c3p4c5")
    c = m.defm.backbone(x)
    assert all(torch.equal(a[k], o[k]) for k in o)
    assert torch.equal(b["P4"], o["P4"])
    ln = lambda t: F.layer_norm(t.permute(0, 2, 3, 1), t.shape[1:2]).permute(0, 3, 1, 2)   # noqa: E731
    assert torch.allclose(b["C3"], ln(c["c3"])) and torch.allclose(b["C5"], ln(c["c5"]))


@torch.no_grad()
def test_p3_p5_constantes_solo_si_el_checkpoint_las_anula(entrada):
    """D-05: con pesos aleatorios P3 y P5 cambian con la profundidad; al anular las convoluciones de salida y la
    fusión de P5 (lo que D-05 describe en el checkpoint), salen constantes y P4 sigue cambiando. La ruta de
    extracción no produce la constancia: la produce el checkpoint."""
    _, p, v = entrada
    defm = construir("E2").defm
    otra = torch.flip(p, (-1,)) * 1.3
    rel = lambda a, b: float((a - b).norm() / a.norm())   # noqa: E731
    o1, o2 = defm(rgbd.entrada_defm(p, v))["dense_bifpn"], defm(rgbd.entrada_defm(otra, v))["dense_bifpn"]
    assert all(rel(o1[k], o2[k]) > 1e-2 for k in ("P3", "P4", "P5"))
    d = copy.deepcopy(defm)
    for i in (0, 2):
        d.fpn.blocks[0].convs[i].weight.mul_(1e-9 / d.fpn.blocks[0].convs[i].weight.norm())
    d.fpn.blocks[0].fuse_bottomup[1].weights.zero_()
    o1, o2 = d(rgbd.entrada_defm(p, v))["dense_bifpn"], d(rgbd.entrada_defm(otra, v))["dense_bifpn"]
    assert rel(o1["P3"], o2["P3"]) < 1e-6 and rel(o1["P5"], o2["P5"]) < 1e-6 and rel(o1["P4"], o2["P4"]) > 1e-2
