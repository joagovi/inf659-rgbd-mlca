# -*- coding: utf-8 -*-
"""Entrena los experimentos de la propuesta RGB-D en NYUv2 (40 clases) y los evalúa en cada condición de la
profundidad (`rgbd.CONDICIONES`).

    python src/entrenar_rgbd.py --humo                       # unos pasos y 20 imágenes: que todo corre
    python src/entrenar_rgbd.py --barrido E1                 # lr en la validación por escenas, semilla 0
    python src/entrenar_rgbd.py --exp E1 --lr 1e-3 --semilla 0   # con las 795, evaluación en el test

Protocolo, igual para todos los experimentos:
- Encoders congelados; se entrenan la fusión y la cabeza de EfficientViT (`rgbd.RGBD`).
- Entrenamiento con la profundidad rellenada (`depths`). Aumento en la GPU: escala U(1, 1.5) con recorte
  aleatorio de 480×640 y volteo horizontal.
- AdamW (decaimiento 0.01), lote 8, 100 pasos de calentamiento y caída polinómica (0.9); bf16.
- Sin selección de época: se evalúa el modelo final. La lr se elige en una validación del 15 % de las escenas de
  entrenamiento (`nyuv2.validacion_por_escenas`), nunca en el test.
- mIoU sobre las 40 clases a 480×640, ignorando «sin etiqueta».

Cada corrida deja en runs/rgbd/<nombre>/ resultado.json y cabeza.pt (solo lo entrenado).
Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
import efficientvit_seg as sg  # noqa: E402

RUNS = sg.RAIZ / "runs" / "rgbd"
FUSION = {"E1": "rgb", "E2": "concat", "E4": "mlca"}
LRS = (3e-4, 1e-3, 3e-3)
DEV = "cuda"
IGNORAR = 255


def lote_a_gpu(datos, idx):
    """Arrays del memmap -> tensores en la GPU: rgb (B,3,H,W) uint8, prof y cruda en mm, etiqueta 0..39 / 255."""
    idx = np.sort(idx)
    rgb = torch.from_numpy(np.ascontiguousarray(datos["rgb"][idx])).to(DEV).permute(0, 3, 1, 2)
    prof = torch.from_numpy(datos["prof"][idx].astype(np.float32)).to(DEV)[:, None]
    et = torch.from_numpy(datos["etiqueta"][idx].astype(np.int64)).to(DEV)
    et = torch.where(et == 0, torch.full_like(et, IGNORAR), et - 1)
    return rgb, prof, et


def aumentar(rgb, prof, et, gen):
    b, _, h, w = rgb.shape
    s = 1 + 0.5 * torch.rand(1, generator=gen).item()
    H, W = round(h * s), round(w * s)
    rgb = F.interpolate(rgb.float(), (H, W), mode="bilinear", align_corners=False)
    prof = F.interpolate(prof, (H, W), mode="nearest")
    et = F.interpolate(et[:, None].float(), (H, W), mode="nearest")[:, 0].long()
    sal = [], [], []
    for i in range(b):
        y = torch.randint(0, H - h + 1, (1,), generator=gen).item()
        x = torch.randint(0, W - w + 1, (1,), generator=gen).item()
        t = [rgb[i, :, y:y + h, x:x + w], prof[i, :, y:y + h, x:x + w], et[i, y:y + h, x:x + w]]
        if torch.rand(1, generator=gen).item() < 0.5:
            t = [z.flip(-1) for z in t]
        for lista, z in zip(sal, t):
            lista.append(z)
    return [torch.stack(z) for z in sal]


def normalizar_rgb(rgb):
    return (rgb.float() / 255 - sg.MEAN.to(DEV)) / sg.STD.to(DEV)


def evaluar(modelo, datos, posiciones, condiciones, lote=4):
    """mIoU por condición. Las condiciones son deterministas por imagen (semilla = índice en el split)."""
    modelo.eval()
    res = {}
    for cond in condiciones:
        conf = torch.zeros(40, 40, dtype=torch.long, device=DEV)
        for k in range(0, len(posiciones), lote):
            idx = posiciones[k:k + lote]
            rgb, _, et = lote_a_gpu(datos, idx)
            pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in np.sort(idx)]
            prof = torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None]
            valida = torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = modelo(normalizar_rgb(rgb), prof, valida)
            pred = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            m = et != IGNORAR
            conf += torch.bincount(40 * et[m] + pred[m], minlength=1600).view(40, 40)
        conf = conf.double()
        iou = (conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)).cpu().numpy()
        res[cond] = {"miou": float(100 * iou.mean()), "acc_pixel": float(100 * conf.diag().sum() / conf.sum()),
                     "iou_clase": [round(float(100 * x), 2) for x in iou]}
        if modelo.fusion == "rgb":                       # sin profundidad, todas las condiciones dan lo mismo
            res = {c: res[cond] for c in condiciones}
            break
    return res


def entrenar(exp, lr, semilla, epocas=60, lote=8, validacion=False, humo=False, nombre=None):
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    tr_datos = nyuv2.cache("entrenamiento")
    if validacion:
        pos_tr, pos_ev = nyuv2.validacion_por_escenas()
        ev_datos, condiciones = tr_datos, ["completa"]
    else:
        pos_tr, ev_datos = np.arange(len(tr_datos["rgb"])), nyuv2.cache("test")
        pos_ev, condiciones = np.arange(len(ev_datos["rgb"])), rgbd.CONDICIONES
    if humo:
        epocas, pos_ev = 1, pos_ev[:20]
    nombre = nombre or f"{'val_' if validacion else ''}{exp}_lr{lr:g}_s{semilla}{'_humo' if humo else ''}"
    salida = RUNS / nombre
    salida.mkdir(parents=True, exist_ok=True)

    modelo = rgbd.RGBD(FUSION[exp]).to(DEV)
    opt = torch.optim.AdamW(modelo.entrenables(), lr=lr, weight_decay=0.01)
    pasos_ep = len(pos_tr) // lote
    total = 5 if humo else epocas * pasos_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * (1 - min(t, total - 1) / total) ** 0.9)
    print(f"{nombre}: {sum(p.numel() for p in modelo.entrenables()) / 1e6:.2f} M entrenables, {total} pasos",
          flush=True)

    t0, paso, historial = time.time(), 0, []
    modelo.train()
    while paso < total:
        orden = pos_tr[torch.randperm(len(pos_tr), generator=gen).numpy()]
        for k in range(0, len(orden) - lote + 1, lote):
            rgb, prof, et = aumentar(*lote_a_gpu(tr_datos, orden[k:k + lote]), gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = modelo(normalizar_rgb(rgb), prof / 1e3, prof > 0)
            lg = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False)
            perdida = F.cross_entropy(lg, et, ignore_index=IGNORAR)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step(); sched.step(); paso += 1
            historial.append(float(perdida))
            if paso % pasos_ep == 0 or paso == total:
                print(f"  paso {paso}/{total}  pérdida {np.mean(historial[-pasos_ep:]):.3f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if paso == total:
                break

    t_ent = time.time() - t0
    res = evaluar(modelo, ev_datos, pos_ev, condiciones)
    out = {"exp": exp, "fusion": FUSION[exp], "lr": lr, "semilla": semilla, "epocas": epocas, "lote": lote,
           "pasos": total, "validacion": validacion, "humo": humo, "minutos_entrenamiento": t_ent / 60,
           "perdida_final": float(np.mean(historial[-pasos_ep:])), "clases": nyuv2.clases40()[1:],
           "evaluado_en": "validación por escenas" if validacion else "test oficial (654)", "resultados": res}
    (salida / "resultado.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    torch.save({k: v for k, v in modelo.state_dict().items()
                if not k.startswith(("evit.backbone", "defm"))}, salida / "cabeza.pt")
    print("  " + "  ".join(f"{c} {r['miou']:.2f}" for c, r in res.items()), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(FUSION))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--barrido", choices=list(FUSION))
    ap.add_argument("--humo", action="store_true")
    a = ap.parse_args()
    if a.humo:
        for exp in ("E1", "E2", "E4"):
            entrenar(exp, 1e-3, 0, humo=True)
    elif a.barrido:
        for lr in LRS:
            if not (RUNS / f"val_{a.barrido}_lr{lr:g}_s0" / "resultado.json").exists():
                entrenar(a.barrido, lr, 0, epocas=a.epocas, validacion=True)
    else:
        entrenar(a.exp, a.lr, a.semilla, epocas=a.epocas)
