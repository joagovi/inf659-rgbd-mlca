# -*- coding: utf-8 -*-
"""Carga de EfficientViT-Seg con sus pesos oficiales y constantes comunes.

EfficientViT se clona en `third_party/efficientvit` (commit de7d773; ver README). Uso de IA: escrito con
asistencia de IA (Claude, Anthropic).
"""
import sys
import types
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
EV_DIR = RAIZ / "third_party" / "efficientvit"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------------------------- modelos
def cargar_efficientvit(variante="b1", dataset="ade20k"):
    """EfficientViT-Seg con sus pesos oficiales, sin instalar las dependencias del repositorio.

    Dos sustituciones en memoria: un Triton vacío mientras se importa, y el paquete
    `efficientvit.models.efficientvit` registrado sin ejecutar su __init__ (que importa SAM y DC-AE).
    """
    from huggingface_hub import hf_hub_download

    class _Comodin(types.ModuleType):
        def __getattr__(s, k):
            if k.startswith("__"):
                raise AttributeError(k)
            return _Comodin(k)

        def __call__(s, *a, **kw):
            return a[0] if (len(a) == 1 and callable(a[0]) and not kw) else _Comodin("llamada")

    rc = types.ModuleType("efficientvit.apps.trainer.run_config")

    class Scheduler:
        PROGRESS = 0  # solo lo usa DropPath, que estos modelos no activan

    rc.Scheduler = Scheduler
    pkg = types.ModuleType("efficientvit.models.efficientvit")
    pkg.__path__ = [str(EV_DIR / "efficientvit" / "models" / "efficientvit")]
    for k, v in {"triton": _Comodin("triton"), "triton.language": _Comodin("triton.language"),
                 "efficientvit.apps.trainer.run_config": rc, "efficientvit.models.efficientvit": pkg}.items():
        sys.modules.setdefault(k, v)
    if str(EV_DIR) not in sys.path:
        sys.path.insert(0, str(EV_DIR))
    try:
        from efficientvit.models.efficientvit import seg as evseg
        from efficientvit.models.nn.norm import set_norm_eps
        from efficientvit.models.utils import load_state_dict_from_file
    finally:
        for k in ("triton", "triton.language"):
            if isinstance(sys.modules.get(k), _Comodin):
                sys.modules.pop(k)
    m = getattr(evseg, f"efficientvit_seg_{variante}")(dataset=dataset)
    set_norm_eps(m, 1e-5)
    ruta = hf_hub_download("han-cai/efficientvit-seg", f"efficientvit_seg_{variante}_{dataset}.pt")
    m.load_state_dict(load_state_dict_from_file(ruta))
    return m
