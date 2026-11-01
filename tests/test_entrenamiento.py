# -*- coding: utf-8 -*-
"""Integración: entrenamiento, evaluación, reanudación, huellas, derivación y combinaciones, en CPU con datos
sintéticos y pesos aleatorios (D-06). Los mIoU no significan nada aquí; lo que se comprueba es el protocolo."""
import json

import numpy as np
import pytest
import torch

import combinar
import entrenar_rgbd as E
import rgbd

pytestmark = [pytest.mark.modelo, pytest.mark.lenta]


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RUNS", tmp_path)
    monkeypatch.setattr(combinar, "RUNS", tmp_path)
    return tmp_path


def cabeza(carpeta):
    return torch.load(carpeta / "cabeza.pt", weights_only=True)


def iguales(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("exp", ["E1", "E2", "E4", "E5", "E2c", "E3c", "E3cd", "E4c", "E4cd", "E4cf", "E1u", "E4ed"])
def test_cada_experimento_entrena_y_evalua(runs, exp):
    out = E.entrenar(exp, 1e-3, 0, epocas=1, pasos_max=2)
    conds = rgbd.CONDICIONES + ([] if E.EXPERIMENTOS[exp].historico else rgbd.CONDICIONES_EXTRA)
    assert list(out["resultados"]) == conds
    assert out["huella"] == E.huella(exp, 1e-3, 0, 1, 8, False, False, None)
    assert out["entorno"]["sin_pesos"] and out["config"]["fusion"] == E.FUSION[exp]
    z = np.load(runs / E.nombre_corrida(exp, 1e-3, 0) / "por_imagen.npz")
    i, u = z["completa_inter"], z["completa_union"]
    assert i.shape == u.shape == (len(z["indices"]), 40)
    miou = 100 * float(np.mean(i.sum(0) / np.maximum(u.sum(0), 1)))
    assert abs(miou - out["resultados"]["completa"]["miou"]) < 1e-9


def test_e3_y_e4_se_entrenan_igual_sin_dropout(runs):
    """Con la profundidad rellenada (m = 1 siempre), las tres máscaras dan los mismos pesos: la máscara solo
    actúa al evaluar. Con dropout, binaria y fraccional también; sin máscara, no."""
    for exp in ("E3c", "E4c", "E4cf", "E3cd", "E4cd", "E4cfd"):
        E.entrenar(exp, 1e-3, 0, epocas=2, pasos_max=4)
    c = {e: cabeza(runs / E.nombre_corrida(e, 1e-3, 0)) for e in ("E3c", "E4c", "E4cf", "E3cd", "E4cd", "E4cfd")}
    assert iguales(c["E3c"], c["E4c"]) and iguales(c["E3c"], c["E4cf"])
    assert iguales(c["E4cd"], c["E4cfd"])
    assert not iguales(c["E3cd"], c["E4cd"])
    assert E.equivalentes_en_entrenamiento("E4c", "E3c") and E.equivalentes_en_entrenamiento("E4cd", "E4cfd")
    assert not E.equivalentes_en_entrenamiento("E4cd", "E3cd") and not E.equivalentes_en_entrenamiento("E4c", "E4cd")


def test_e3e_y_e4e_se_entrenan_igual_sin_dropout(runs):
    """D-07: lo mismo con la MLCA del codificador, que se entrena junto con la última etapa del color."""
    for exp in ("E3e", "E4e"):
        E.entrenar(exp, 1e-3, 0, epocas=2, pasos_max=4)
    a, b = (cabeza(runs / E.nombre_corrida(e, 1e-3, 0)) for e in ("E3e", "E4e"))
    assert iguales(a, b)
    assert any(k.startswith("fus_cod.") for k in a) and any(k.startswith("evit.backbone.stages.3.") for k in a)


def _lotes(monkeypatch, exp):
    vistos = []
    original = E.lote_a_gpu

    def espia(datos, idx):
        vistos.append(tuple(np.sort(idx)))
        return original(datos, idx)
    monkeypatch.setattr(E, "lote_a_gpu", espia)
    E.entrenar(exp, 1e-3, 0, epocas=3, pasos_max=6)
    monkeypatch.setattr(E, "lote_a_gpu", original)
    return vistos[:6]                                  # los del entrenamiento (luego vienen los de la evaluación)


def test_el_dropout_no_cambia_el_orden_de_los_datos(runs, monkeypatch):
    """D-06: en la familia «c» el dropout usa su propio generador. En la v3 (E5) usaba el de los datos."""
    assert _lotes(monkeypatch, "E4c") == _lotes(monkeypatch, "E4cd")
    assert _lotes(monkeypatch, "E4") != _lotes(monkeypatch, "E5")


def test_reanudar_da_el_mismo_resultado(runs, tmp_path, monkeypatch):
    E.entrenar("E4cd", 1e-3, 0, epocas=4, pasos_max=7, guardar_cada=0, nombre="seguida")
    with pytest.raises(KeyboardInterrupt):
        E.entrenar("E4cd", 1e-3, 0, epocas=4, pasos_max=7, guardar_cada=1000, cortar_en=3, nombre="cortada")
    assert (runs / "cortada" / "estado.pt").exists() and not (runs / "cortada" / "resultado.json").exists()
    out = E.entrenar("E4cd", 1e-3, 0, epocas=4, pasos_max=7, guardar_cada=1000, nombre="cortada")
    assert iguales(cabeza(runs / "seguida"), cabeza(runs / "cortada"))
    a = json.loads((runs / "seguida" / "resultado.json").read_text(encoding="utf-8"))
    assert a["resultados"] == out["resultados"] and a["perdida_final"] == out["perdida_final"]
    assert (runs / "cortada" / "estado.pt").exists()          # nunca se borra


def test_huella_distinta_no_se_reutiliza_ni_se_pisa(runs):
    E.entrenar("E4c", 1e-3, 0, epocas=1, pasos_max=2)
    ruta = runs / E.nombre_corrida("E4c", 1e-3, 0) / "resultado.json"
    out = json.loads(ruta.read_text(encoding="utf-8"))
    out["huella"] = "otra"
    ruta.write_text(json.dumps(out), encoding="utf-8")
    with pytest.raises(E.ConfigConflict):
        E.entrenar("E4c", 1e-3, 0, epocas=1, pasos_max=2)
    assert json.loads(ruta.read_text(encoding="utf-8"))["huella"] == "otra"


def test_un_resultado_sin_huella_solo_vale_en_los_historicos(runs):
    for exp, vale in (("E4", True), ("E4c", False)):
        d = runs / E.nombre_corrida(exp, 1e-3, 0)
        d.mkdir()
        (d / "resultado.json").write_text(json.dumps({"exp": exp, "resultados": {}}), encoding="utf-8")
        if vale:
            assert E.resultado_valido(d, exp, "x")
        else:
            with pytest.raises(E.ConfigConflict):
                E.resultado_valido(d, exp, "x")


def test_un_error_queda_registrado_y_se_propaga(runs, monkeypatch):
    def falla(*a, **k):
        raise RuntimeError("fallo provocado")
    monkeypatch.setattr(E, "ce", falla)
    with pytest.raises(RuntimeError):
        E.entrenar("E2c", 1e-3, 0, epocas=1, pasos_max=2)
    d = runs / E.nombre_corrida("E2c", 1e-3, 0)
    assert "fallo provocado" in (d / "error.log").read_text(encoding="utf-8")
    assert not (d / "resultado.json").exists()


def test_completo_y_derivar_e3_de_e4(runs):
    E.completo("E4c", pasos_max=2)
    (lr, _), val = E.elegida("E4c")
    E.derivar("E4c", "E3c")
    (lr3, _), val3 = E.elegida("E3c")
    assert lr3 == lr and val3 == val
    for s in range(3):
        a = json.loads((runs / E.nombre_corrida("E4c", lr, s) / "resultado.json").read_text(encoding="utf-8"))
        b = json.loads((runs / E.nombre_corrida("E3c", lr, s) / "resultado.json").read_text(encoding="utf-8"))
        assert b["derivado_de"] == E.nombre_corrida("E4c", lr, s)
        assert a["resultados"]["completa"] == b["resultados"]["completa"]      # m = 1: la misma red
        modelo, _ = E.cargar(runs / E.nombre_corrida("E3c", lr, s))
        assert modelo.mascara == "ninguna"
    with pytest.raises(ValueError):
        E.derivar("E4cd", "E3cd")
    E.derivar("E4c", "E3c")                                                    # ya está: no repite nada


def test_mezcla_respaldo_exacto_y_mismo_calculo_que_evaluar(runs):
    p1, p2 = torch.rand(2, 40, 12, 16).softmax(1), torch.rand(2, 40, 12, 16).softmax(1)
    assert torch.equal(combinar.mezclar(p1, p2, torch.zeros(2, 1, 12, 16)), p1)
    E.entrenar("E1", 1e-3, 0, epocas=1, pasos_max=2)
    m, out = E.cargar(runs / E.nombre_corrida("E1", 1e-3, 0))
    datos = E.nyuv2.cache("test")
    r = combinar.recorrer(m, m, datos, np.arange(len(datos["rgb"])), lote=2, interpolar="logits")
    assert abs(r["completa"]["color"] - out["resultados"]["completa"]["miou"]) < 1e-9


def test_main_devuelve_cero_y_lista(capsys):
    assert E.main(["--listar"]) == 0
    assert "E4cd" in capsys.readouterr().out
