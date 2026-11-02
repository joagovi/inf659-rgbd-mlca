# -*- coding: utf-8 -*-
"""Procedencia de los pesos preentrenados y del entorno (D-06).

Fija de dónde salen los dos extractores congelados y deja constancia de qué se usó en cada corrida:

- DeFM: el código se carga de `leggedrobotics/defm` en un commit fijo (`DEFM_COMMIT`), no de `main`. Desde el
  22/10/2026 (`b5538c8`) `main` lee la configuración de la arquitectura de Hugging Face al cargar los pesos; con
  `pretrained_path` se usa la configuración empaquetada en el commit. El código de `defm/models/` no cambia
  entre `63ec5e1` (27/04/2026) y `4849a36` (26/10/2026), así que cualquier commit de ese tramo construye la
  misma red que usaron las corridas de la v3.
- Los checkpoints (`defm_resnet18.pth`, `efficientvit_seg_b1_ade20k.pt`) se descargan una vez de Hugging Face
  (revisión opcionalmente fijada) o se leen de una ruta local, y se registra su SHA-256. Si `SHA256_ESPERADO`
  tiene un valor, la carga falla cuando el archivo no coincide.

Variables de entorno (todas opcionales):

    RGBD_DEFM_REPO     carpeta con un clon local de leggedrobotics/defm (en lugar de torch.hub desde GitHub)
    RGBD_DEFM_PESOS    ruta local de defm_resnet18.pth
    RGBD_EVIT_PESOS    ruta local de efficientvit_seg_b1_ade20k.pt
    RGBD_SIN_PESOS=1   construye las dos redes con pesos aleatorios: solo para pruebas y medidas de coste
                       (las corridas guardan `sin_pesos: true` y `tablas.py` no las mezcla con las reales)

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DEFM_COMMIT = "4849a368d669ae5d34bcd612fa3620f0e8a7ab83"      # leggedrobotics/defm, 2026-10-26
# Revisiones de Hugging Face y SHA-256 de los checkpoints que usaron todas las corridas de la v3 a D-08 (los mismos
# en cada resultado.json; fijados el 2026-11-02). Con None se baja la revisión que haya y no se comprueba nada.
DEFM_HF_REPO, DEFM_HF_REVISION = "leggedrobotics/defm", "b6c9eb68ac833ccd93bf2571279235cb9b065251"
EVIT_COMMIT = "de7d773"                                         # third_party/efficientvit (README)
EVIT_HF_REPO, EVIT_HF_REVISION = "han-cai/efficientvit-seg", "cf3ccaf9cbaf670a2cb283612773546824ca0aa1"
SHA256_ESPERADO = {"defm_resnet18.pth": "781d61dda55a789f445f23445cf7cba5d83693e3cc002281245240c523b56a44",
                   "efficientvit_seg_b1_ade20k.pt": "59cca6a7ff4d1a6494b7f12072c6a68fee40ef401e971f26af01a2e0f3c8f5e7"}

_REGISTRO = {}          # archivo -> {ruta, sha256, bytes}; lo leen las corridas para su resultado.json


def sin_pesos():
    return os.environ.get("RGBD_SIN_PESOS", "") not in ("", "0")


def pesos_aleatorios_fijos(modelo, construir, semilla):
    """Sin pesos (pruebas): sustituye la inicialización de `modelo` por la de otra construcción con una semilla
    fija, sin tocar el generador global. Así un modelo recargado tiene los mismos codificadores «congelados» que el
    entrenado, y el generador global se consume igual que con los pesos oficiales."""
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(semilla)
        ref = construir()
    modelo.load_state_dict(ref.state_dict())
    return modelo


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def registrar(nombre, ruta):
    """Registra el checkpoint usado y comprueba su SHA-256 si se conoce el esperado."""
    ruta = Path(ruta)
    info = {"ruta": str(ruta), "sha256": sha256(ruta), "bytes": ruta.stat().st_size}
    esperado = SHA256_ESPERADO.get(nombre)
    if esperado and esperado != info["sha256"]:
        raise RuntimeError(f"{nombre}: SHA-256 {info['sha256']} distinto del esperado {esperado}")
    _REGISTRO[nombre] = info
    return info


def ruta_pesos(nombre, variable, repo, revision):
    """Ruta local del checkpoint: la de la variable de entorno o la caché de Hugging Face."""
    ruta = os.environ.get(variable)
    if ruta:
        return ruta
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, nombre, revision=revision)


def commit_git(carpeta):
    try:
        return subprocess.run(["git", "-C", str(carpeta), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - sin git o sin repositorio
        return None


def entorno():
    """Lo que se guarda en cada resultado.json para poder repetir la corrida."""
    import numpy as np
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": gpu, "plataforma": platform.platform(), "repo_commit": commit_git(RAIZ),
            "defm_commit": os.environ.get("RGBD_DEFM_REPO") and commit_git(os.environ["RGBD_DEFM_REPO"]) or DEFM_COMMIT,
            "efficientvit_commit": EVIT_COMMIT, "pesos": dict(_REGISTRO), "sin_pesos": sin_pesos()}


if __name__ == "__main__":
    print(json.dumps(entorno(), indent=1, ensure_ascii=False))
