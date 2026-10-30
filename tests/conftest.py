# -*- coding: utf-8 -*-
"""Configuración común de las pruebas (D-06). Se fija el entorno ANTES de importar el código del proyecto:

- datos sintéticos (`nyuv2.datos_sinteticos`) y pesos aleatorios: las pruebas no necesitan NYUv2 ni descargar
  pesos; las corridas van a una carpeta temporal, nunca a runs/rgbd;
- CPU, salvo las pruebas marcadas `gpu`, que se saltan si no hay CUDA;
- DeFM: el clon de RGBD_DEFM_REPO si existe; si no, torch.hub desde GitHub en el commit fijado.

    pytest -q                 # todas (unos minutos en CPU)
    pytest -q -m "not lenta"  # sin las de entrenamiento

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RGBD_DATOS_SINTETICOS", "1")
os.environ.setdefault("RGBD_SIN_PESOS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("RGBD_DEV", "cpu")
os.environ.setdefault("RGBD_RUNS", tempfile.mkdtemp(prefix="rgbd_pruebas_"))
os.environ.setdefault("PYTHONWARNINGS", "ignore")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def pytest_configure(config):
    config.addinivalue_line("markers", "lenta: entrena unos pasos (segundos a minutos en CPU)")
    config.addinivalue_line("markers", "gpu: necesita CUDA")
    config.addinivalue_line("markers", "modelo: necesita el código de EfficientViT (third_party) y de DeFM")


def pytest_collection_modifyitems(config, items):
    sin_cuda = pytest.mark.skip(reason="no hay CUDA")
    sin_ev = pytest.mark.skip(reason="falta third_party/efficientvit")
    for it in items:
        if "gpu" in it.keywords and not torch.cuda.is_available():
            it.add_marker(sin_cuda)
        if "modelo" in it.keywords and not (RAIZ / "third_party" / "efficientvit").exists():
            it.add_marker(sin_ev)
