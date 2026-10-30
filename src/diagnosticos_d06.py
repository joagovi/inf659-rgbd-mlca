# -*- coding: utf-8 -*-
"""Diagnósticos de MLCA (E4, E5 y la familia «c»), sin reentrenar (D-06). Exploratorios: no sirven para elegir.

    python src/diagnosticos_d06.py --exps E4 E5 --datos validacion     # por defecto: la validación por escenas
    python src/diagnosticos_d06.py --exps E4c E3c E4cd E3cd --datos test   # marca el resultado como exploratorio

Por defecto se usa la validación por escenas con los modelos de validación (`val_*`), para no seguir gastando el
test. Para cada modelo:

1. Tamaño relativo de lo que suma la fusión en cada escala, ‖Δ‖/‖f_color‖ (Δ = f_fusión − f_color), y qué parte
   es un sesgo constante: ‖Δ − media(Δ)‖/‖Δ‖ sobre posiciones e imágenes (0 = la fusión suma una constante, como
   pasa en 1/8 y 1/32 con P3 y P5 de la BiFPN).
2. Dependencia de la cabeza de cada escala: mIoU con la profundidad completa quitando Δ en una escala, o en todas.
3. Sin profundidad: concordancia por píxel de la predicción con la de un E1 (o E1u) de la misma semilla, si
   existe, y la proporción de píxeles en las tres clases más predichas (colapso).
4. Estabilidad de la atención: distribución de φ(q)ᵀz por cabeza (mínimo, proporción < 1e-3) y proporción de
   consultas anuladas por la ReLU.
5. Gradientes: norma del gradiente de Q, K, V y la proyección de salida en unos lotes de entrenamiento con los
   pesos finales.

Resultado en runs/rgbd/diagnosticos_d06_<datos>.json. Uso de IA: escrito con asistencia de IA (Claude,
Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
from entrenar_rgbd import (DEV, IGNORAR, RUNS, autocast, cargar, ce, elegida, lote_a_gpu,  # noqa: E402
                           nombre_corrida, normalizar_rgb)

ESC = {"stage2": "1/8", "stage3": "1/16", "stage4": "1/32"}


def miou(conf):
    c = conf.double()
    return float(100 * (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)).mean())


def lotes(datos, pos, cond, lote=4):
    for k in range(0, len(pos), lote):
        idx = np.sort(pos[k:k + lote])
        rgb, _, et = lote_a_gpu(datos, idx)
        pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in idx]
        yield (normalizar_rgb(rgb), torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None],
               torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None], et)


def sumar(conf, lg, et):
    pred = F.interpolate(lg.float(), et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
    m = et != IGNORAR
    conf += torch.bincount(40 * et[m] + pred[m], minlength=1600).view(40, 40)
    return pred


@torch.no_grad()
def por_dentro(modelo, datos, pos, ref=None):
    out = {}
    for cond in ("completa", "cruda", "regiones", "sin"):
        rel, suma, suma2, cnt, qz_min, qz_peq, q_nula = {}, {}, {}, {}, [], [], []
        quitar = {"ninguna": (), **{f"sin_{e}": (c,) for c, e in ESC.items()}, "sin_todas": tuple(ESC)}
        conf = {v: torch.zeros(40, 40, dtype=torch.long, device=DEV) for v in quitar}
        acuerdo, total, cuentas = 0, 0, torch.zeros(40, device=DEV)
        for x, p, v, et in lotes(datos, pos, cond):
            with autocast():
                s = modelo(x, p, v, rasgos=True)
            for c in s["f_color"]:
                d = (s["f_fusion"][c] - s["f_color"][c]).float()
                rel.setdefault(c, []).extend((d.flatten(1).norm(dim=1) / s["f_color"][c].float().flatten(1).norm(dim=1)).tolist())
                suma[c] = suma.get(c, 0) + d.sum((0, 2, 3))
                suma2[c] = suma2.get(c, 0) + d.pow(2).sum((0, 2, 3))
                cnt[c] = cnt.get(c, 0) + d.shape[0] * d.shape[2] * d.shape[3]
            for var, qs in quitar.items():
                g = {**{k: s["f_color"][k] if k in qs else s["f_fusion"][k] for k in s["f_color"]}}
                f = modelo.color(x) if var != "ninguna" else None
                if f is not None:
                    f.update(g)
                    lg = rgbd.salida(modelo.evit.head, f)
                else:
                    lg = s["seg"]
                pred = sumar(conf[var], lg, et)
                if var == "ninguna":
                    cuentas += torch.bincount(pred.flatten(), minlength=40)[:40]
                    if ref is not None:
                        with autocast():
                            pr = F.interpolate(ref(x, p, v).float(), et.shape[-2:], mode="bilinear",
                                               align_corners=False).argmax(1)
                        acuerdo += int((pr == pred).sum())
                        total += pred.numel()
            if modelo.fusion == "mlca":                       # φ(q)ᵀz por cabeza en cada escala
                for c, d_ in modelo.pares:
                    blq = modelo.fus[c]
                    fd = rgbd.rasgos_defm(modelo.defm, p, v, modelo.rasgos_prof)[d_].float()
                    q = F.relu(blq._tokens(blq.q(s["f_color"][c].float())))
                    k = F.relu(blq._tokens(blq.k(fd)))
                    mj = s["m"][c].reshape(k.shape[0], 1, -1, 1)
                    k = torch.where(mj > 0, k * mj, torch.zeros_like(k))
                    qz = (q @ k.sum(-2).unsqueeze(-1)).squeeze(-1)
                    qz_min.append(float(qz.min()))
                    qz_peq.append(float((qz < 1e-3).float().mean()))
                    q_nula.append(float((q.sum(-1) == 0).float().mean()))
        tot = int(sum(conf["ninguna"].sum(1)))
        top3 = float(cuentas.sort(descending=True).values[:3].sum() / cuentas.sum().clamp(min=1))
        out[cond] = {
            "miou": {v: miou(c) for v, c in conf.items()},
            "relativo": {ESC[c]: float(np.mean(r)) for c, r in rel.items()},
            # ‖Δ − media(Δ)‖ / ‖Δ‖ sobre todas las posiciones e imágenes: 0 si la fusión suma una constante
            "parte_no_constante": {ESC[c]: float(((suma2[c] - suma[c].pow(2) / cnt[c]).sum()
                                                  / suma2[c].sum().clamp(min=1e-30)).clamp(min=0).sqrt())
                                   for c in suma},
            "acuerdo_con_referencia": acuerdo / total if total else None, "pixeles_top3_clases": top3,
            "phi_q_z_min": min(qz_min) if qz_min else None,
            "phi_q_z_menor_1e-3": float(np.mean(qz_peq)) if qz_peq else None,
            "consultas_nulas": float(np.mean(q_nula)) if q_nula else None, "pixeles": tot}
    return out


def gradientes(modelo, datos, pos, n_lotes=4, lote=4):
    modelo.train()
    normas = {}
    for k in range(n_lotes):
        idx = np.sort(pos[k * lote:(k + 1) * lote])
        rgb, prof, et = lote_a_gpu(datos, idx)
        modelo.zero_grad(set_to_none=True)
        with autocast():
            lg = modelo(normalizar_rgb(rgb), prof / 1e3, prof > 0)
        ce(lg, et).backward()
        for c, _ in modelo.pares:
            for parte in ("q", "k", "v", "salida"):
                if hasattr(modelo.fus[c], parte):
                    g = getattr(modelo.fus[c], parte).weight.grad
                    normas.setdefault(f"{ESC[c]}_{parte}", []).append(0.0 if g is None else float(g.norm()))
    modelo.eval()
    return {k: float(np.mean(v)) for k, v in normas.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["E4", "E5"])
    ap.add_argument("--datos", choices=("validacion", "test"), default="validacion")
    ap.add_argument("--referencia", default="E1", help="modelo de color con el que comparar sin profundidad")
    a = ap.parse_args()
    val = a.datos == "validacion"
    tr = nyuv2.cache("entrenamiento")
    if val:
        datos, pos = tr, nyuv2.validacion_por_escenas()[1]
        semillas = [0]
    else:
        datos = nyuv2.cache("test")
        pos, semillas = np.arange(len(datos["rgb"])), [0, 1, 2]
    out = {"datos": a.datos, "nota": "exploratorio (D-06)" + ("" if val else "; usa el test")}
    for exp in a.exps:
        (lr, _), _ = elegida(exp)
        out[exp] = []
        for s in semillas:
            carpeta = RUNS / nombre_corrida(exp, lr, s, val)
            if not (carpeta / "cabeza.pt").exists() and "derivado_de" not in json.loads(
                    (carpeta / "resultado.json").read_text(encoding="utf-8")):
                print(f"{carpeta.name}: sin cabeza.pt, se salta", flush=True)
                continue
            modelo, _ = cargar(carpeta)
            ref = None
            try:
                (lr_r, _), _ = elegida(a.referencia)
                ref, _ = cargar(RUNS / nombre_corrida(a.referencia, lr_r, s, val))
                ref.eval()
            except (FileNotFoundError, KeyError):
                pass
            r = {"semilla": s, "por_dentro": por_dentro(modelo.eval(), datos, pos, ref),
                 "gradientes": gradientes(modelo, tr, nyuv2.validacion_por_escenas()[0]) if modelo.defm else {}}
            out[exp].append(r)
            print(exp, s, json.dumps(r["por_dentro"]["sin"]["miou"]), flush=True)
    (RUNS / f"diagnosticos_d06_{a.datos}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
