# -*- coding: utf-8 -*-
"""Entrena los experimentos de la propuesta RGB-D en NYUv2 (40 clases) y los evalúa en cada condición de la
profundidad (`rgbd.CONDICIONES`).

    python src/entrenar_rgbd.py --humo [E2u E9]              # unos pasos y 20 imágenes: que todo corre
    python src/entrenar_rgbd.py --barrido E1                 # lr en la validación por escenas, semilla 0
    python src/entrenar_rgbd.py --exp E1 --lr 1e-3 --semilla 0   # con las 795, evaluación en el test
    python src/entrenar_rgbd.py --completo E8                # barrido, elección y 3 semillas
    python src/entrenar_rgbd.py --reevaluar                  # condiciones nuevas en modelos ya entrenados

Protocolo, igual para todos los experimentos:
- Encoders congelados; se entrenan la fusión y la cabeza de EfficientViT (`rgbd.RGBD`). En E1u, E7, E2u y E9
  también la última etapa del codificador de color, con una lr 10 veces menor (D-03, D-04). En E9 el experto de
  color es el E1u ya entrenado de la misma semilla (en la validación, el de la validación), congelado (D-04).
- Entrenamiento con la profundidad rellenada (`depths`); en E5, el 30 % de las muestras sin profundidad; en E8,
  E2h y E9, huecos por muestra (`rgbd.huecos_entrenamiento`). Aumento en la GPU: escala U(1, 1.5) con recorte
  aleatorio de 480×640 y volteo horizontal.
- AdamW (decaimiento 0.01), lote 8, 100 pasos de calentamiento y caída polinómica (0.9); bf16.
- Sin selección de época: se evalúa el modelo final. La lr se elige en una validación del 15 % de las escenas de
  entrenamiento (`nyuv2.validacion_por_escenas`), nunca en el test. En E7 se usa la lr elegida para E1u y se
  elige así el peso de la pérdida auxiliar.
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
FUSION = {"E1": "rgb", "E2": "concat", "E4": "mlca", "E5": "mlca",
          "E1u": "rgb", "E7": "rgb", "E8": "respaldo", "E2h": "concat",
          "E2u": "concat", "E9": "respaldo"}
DROPOUT = {"E5": 0.3}       # modality dropout: proporción de muestras que se entrenan sin ninguna profundidad
HUECOS = {"E8", "E2h", "E9"}    # huecos al entrenar (rgbd.huecos_entrenamiento); E2h es el control de E8
ENTRENABLE = {e: ("stage4",) for e in ("E1u", "E7", "E2u", "E9")}   # etapas del codificador de color que se ajustan
FIJO = {"E9": "E1u"}        # experto de color ya entrenado y congelado (D-04)
FACTOR_COD = 0.1            # lr de esas etapas = 0.1 × la de la cabeza
AUX = {"E7": (1.0, 3.0, 10.0)}   # pesos de la pérdida auxiliar de profundidad que se barren en la validación
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


def ce(lg, et):
    """Entropía cruzada a la resolución de la etiqueta; 0 si el lote no tiene ningún píxel etiquetado."""
    if not (et != IGNORAR).any():
        return lg.sum() * 0
    lg = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(lg, et, ignore_index=IGNORAR)


def perdida_prof(pred, prof):
    """Error cuadrático del log de la profundidad, invariante a la escala de cada imagen (Eigen et al., 2014):
    var(log p̂ − log p) en las celdas de 1/8 con más de la mitad de los píxeles medidos. Invariante a la escala
    porque el aumento amplía la imagen sin cambiar la profundidad. pred (B,1,h,w); prof (B,1,H,W) en mm."""
    tam = pred.shape[-2:]
    v = (prof > 0).float()
    num, den = F.adaptive_avg_pool2d(prof * v, tam), F.adaptive_avg_pool2d(v, tam)
    ok = den > 0.5
    obj = torch.log((num / den.clamp_min(1e-6)).clamp_min(1.0) / 1e3)
    g = torch.where(ok, pred.float() - obj, torch.zeros_like(obj))
    n = ok.sum((1, 2, 3)).clamp_min(1)
    m1, m2 = g.sum((1, 2, 3)) / n, (g ** 2).sum((1, 2, 3)) / n
    return (m2 - m1 ** 2).mean()


def construir(exp):
    return rgbd.RGBD(FUSION[exp], entrenable=ENTRENABLE.get(exp, ()), aux_prof=exp in AUX,
                     experto_fijo=exp in FIJO).to(DEV)


def experto_de(exp, semilla, validacion):
    """Carpeta del modelo que hace de experto de color fijo (E9): el de `FIJO[exp]` con su lr elegida y la misma
    semilla; en la validación, su corrida de validación, que no vio esas escenas."""
    base = FIJO[exp]
    (lr, _), _ = elegida(base)
    return RUNS / nombre_corrida(base, lr, 0 if validacion else semilla, validacion)


def nombre_corrida(exp, lr, semilla, validacion=False, aux=None, humo=False):
    return (f"{'val_' if validacion else ''}{exp}_lr{lr:g}{'' if aux is None else f'_aux{aux:g}'}"
            f"_s{semilla}{'_humo' if humo else ''}")


def cargar(carpeta):
    """Modelo final de una corrida (runs/rgbd/<nombre>/): se construye según su resultado.json y carga cabeza.pt."""
    out = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
    modelo = construir(out["exp"])
    info = modelo.load_state_dict(torch.load(carpeta / "cabeza.pt", map_location=DEV), strict=False)
    assert not info.unexpected_keys and all(k.startswith(("evit.backbone", "defm")) for k in info.missing_keys)
    return modelo, out


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


def entrenar(exp, lr, semilla, epocas=60, lote=8, validacion=False, humo=False, nombre=None, aux=None):
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    gen = torch.Generator().manual_seed(semilla)
    if exp in AUX and aux is None:
        aux = AUX[exp][0]
    tr_datos = nyuv2.cache("entrenamiento")
    if validacion:
        pos_tr, pos_ev = nyuv2.validacion_por_escenas()
        ev_datos, condiciones = tr_datos, ["completa"]
    else:
        pos_tr, ev_datos = np.arange(len(tr_datos["rgb"])), nyuv2.cache("test")
        pos_ev, condiciones = np.arange(len(ev_datos["rgb"])), rgbd.CONDICIONES
    if humo:
        epocas, pos_ev = 1, pos_ev[:20]
    nombre = nombre or nombre_corrida(exp, lr, semilla, validacion, aux, humo)
    salida = RUNS / nombre
    salida.mkdir(parents=True, exist_ok=True)

    modelo = construir(exp)
    if exp in FIJO:
        experto = experto_de(exp, semilla, validacion)
        modelo.fijar_experto(torch.load(experto / "cabeza.pt", map_location=DEV))
    nombres = [n for n, p in modelo.named_parameters() if p.requires_grad]
    cod = [p for n, p in modelo.named_parameters() if p.requires_grad and n.startswith("evit.backbone")]
    resto = [p for n, p in modelo.named_parameters() if p.requires_grad and not n.startswith("evit.backbone")]
    grupos = [{"params": resto}] + ([{"params": cod, "lr": lr * FACTOR_COD}] if cod else [])
    opt = torch.optim.AdamW(grupos, lr=lr, weight_decay=0.01)
    pasos_ep = len(pos_tr) // lote
    total = 5 if humo else epocas * pasos_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * (1 - min(t, total - 1) / total) ** 0.9)
    print(f"{nombre}: {sum(p.numel() for p in modelo.entrenables()) / 1e6:.2f} M entrenables "
          f"({sum(p.numel() for p in cod) / 1e6:.2f} M del codificador), {total} pasos", flush=True)

    t0, paso, historial = time.time(), 0, []
    modelo.train()
    while paso < total:
        orden = pos_tr[torch.randperm(len(pos_tr), generator=gen).numpy()]
        for k in range(0, len(orden) - lote + 1, lote):
            rgb, prof, et = aumentar(*lote_a_gpu(tr_datos, orden[k:k + lote]), gen)
            if exp in DROPOUT:                          # la muestra pierde toda la profundidad, como «sin»
                quita = torch.rand(len(prof), generator=gen) < DROPOUT[exp]
                prof = torch.where(quita.to(DEV)[:, None, None, None], torch.zeros_like(prof), prof)
            if exp in HUECOS:
                prof = rgbd.huecos_entrenamiento(prof, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sal = modelo(normalizar_rgb(rgb), prof / 1e3, prof > 0, todo=True)
            perdida = ce(sal["seg"], et)
            if "rgb" in sal:      # E8: cada experto con su pérdida; el de profundidad, solo donde hay medida
                et_fus = torch.where(prof[:, 0] > 0, et, torch.full_like(et, IGNORAR))
                if exp in FIJO:   # E9: el experto de color no se entrena
                    perdida = perdida + ce(sal["fus"], et_fus)
                else:
                    perdida = perdida + ce(sal["rgb"], et) + ce(sal["fus"], et_fus)
            if "prof" in sal:     # E7: la profundidad solo enseña
                perdida = perdida + aux * perdida_prof(sal["prof"], prof)
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
    if aux is not None:
        out["aux"] = aux
    if exp in ENTRENABLE:
        out["entrenable"] = list(ENTRENABLE[exp])
    if exp in FIJO:
        out["experto"] = experto.name
    (salida / "resultado.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    torch.save({k: v for k, v in modelo.state_dict().items()
                if not k.startswith(("evit.backbone", "defm")) or k in nombres}, salida / "cabeza.pt")
    print("  " + "  ".join(f"{c} {r['miou']:.2f}" for c, r in res.items()), flush=True)
    return out


def elegida(exp):
    """(lr, aux) con mayor mIoU en la validación por escenas (profundidad completa), y todas las cifras."""
    val = {}
    for ruta in RUNS.glob(f"val_{exp}_lr*_s0/resultado.json"):
        out = json.loads(ruta.read_text(encoding="utf-8"))
        val[(out["lr"], out.get("aux"))] = out["resultados"]["completa"]["miou"]
    if not val:
        raise FileNotFoundError(f"no hay barrido de {exp} en {RUNS}")
    return max(val, key=val.get), val


def completo(exp):
    """Barrido en la validación por escenas, elección y 3 semillas en el test. En E7 la lr es la elegida para E1u
    y lo que se barre es el peso de la pérdida auxiliar."""
    if exp in AUX:
        (lr, _), _ = elegida("E1u")
        opciones = [(lr, a) for a in AUX[exp]]
    else:
        opciones = [(lr, None) for lr in LRS]
    for lr, a in opciones:
        if not (RUNS / nombre_corrida(exp, lr, 0, True, a) / "resultado.json").exists():
            entrenar(exp, lr, 0, validacion=True, aux=a)
    (lr, a), val = elegida(exp)
    print(f"{exp}: validación " + ", ".join(f"lr {l:g}{'' if x is None else f' aux {x:g}'}: {v:.2f}"
                                            for (l, x), v in sorted(val.items(), key=str)), flush=True)
    print(f"{exp}: elegida lr {lr:g}" + ("" if a is None else f", aux {a:g}"), flush=True)
    for s in range(3):
        if not (RUNS / nombre_corrida(exp, lr, s, False, a) / "resultado.json").exists():
            entrenar(exp, lr, s, aux=a)


def reevaluar():
    """Evalúa los modelos finales ya entrenados (runs/rgbd/E*_s*) en las condiciones que les falten, sin
    reentrenar: carga cabeza.pt y añade los resultados a su resultado.json."""
    for ruta in sorted(RUNS.glob("E*_lr*_s[0-9]/resultado.json")):
        out = json.loads(ruta.read_text(encoding="utf-8"))
        faltan = [c for c in rgbd.CONDICIONES if c not in out["resultados"]]
        if not faltan:
            continue
        modelo, _ = cargar(ruta.parent)
        datos = nyuv2.cache("test")
        nuevos = evaluar(modelo, datos, np.arange(len(datos["rgb"])), faltan)
        out["resultados"].update({c: nuevos[c] for c in faltan})
        ruta.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(ruta.parent.name, "  ".join(f"{c} {nuevos[c]['miou']:.2f}" for c in faltan), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(FUSION))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--aux", type=float, default=None)
    ap.add_argument("--barrido", choices=list(FUSION))
    ap.add_argument("--completo", choices=list(FUSION), nargs="+")
    ap.add_argument("--humo", choices=list(FUSION), nargs="*", help="sin nombres, todos los experimentos")
    ap.add_argument("--reevaluar", action="store_true")
    a = ap.parse_args()
    if a.reevaluar:
        reevaluar()
    elif a.humo is not None:
        for exp in a.humo or FUSION:
            entrenar(exp, 1e-3, 0, humo=True)
    elif a.completo:
        for exp in a.completo:
            completo(exp)
    elif a.barrido:
        for lr in LRS:
            if not (RUNS / nombre_corrida(a.barrido, lr, 0, True) / "resultado.json").exists():
                entrenar(a.barrido, lr, 0, epocas=a.epocas, validacion=True)
    else:
        entrenar(a.exp, a.lr, a.semilla, epocas=a.epocas, aux=a.aux)
