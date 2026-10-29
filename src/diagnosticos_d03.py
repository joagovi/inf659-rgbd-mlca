# -*- coding: utf-8 -*-
"""Diagnósticos de los resultados de D-03, sobre el test y sin reentrenar (D-04). Son exploratorios: nada de lo
que aquí se mide sirve para elegir un modelo.

1. E7: ¿aprendió profundidad su cabeza auxiliar? Error del log de la profundidad en las celdas de 1/8 con más de
   la mitad de los píxeles medidos, tras alinear la escala de cada imagen (la pérdida es invariante a la escala):
   SI-RMSE del log, AbsRel y δ < 1.25. Referencia: predecir una constante.
2. E8 con profundidad completa: la mezcla, cada experto por separado y el α medio de la compuerta.
3. Promedios de dos modelos ya entrenados, con profundidad completa: dos de color de semillas distintas (E1 + E1,
   E1u + E1u), color ajustado más congelado (E1u + E1) y color más fusión (E1 + E2, E1u + E2), con α fijo.
4. Por clase (de los resultado.json, sin GPU): ganancia de la profundidad (E2 − E1) y del ajuste (E1u − E1).

    python src/diagnosticos_d03.py

Resultado en runs/rgbd/diagnosticos_d03.json.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyuv2  # noqa: E402
from entrenar_rgbd import DEV, IGNORAR, RUNS, cargar, lote_a_gpu, nombre_corrida, normalizar_rgb  # noqa: E402

LR = {"E1": 1e-3, "E1u": 1e-3, "E2": 3e-4, "E8": 3e-3, "E7": 1e-3}     # elegidas en la validación (DECISIONES.md)
ALFAS = (0.5, 0.6, 0.7)
PARES = ((0, 1), (1, 2), (0, 2))


def corrida(exp, s):
    return RUNS / nombre_corrida(exp, LR[exp], s, aux=1.0 if exp == "E7" else None)


def miou(conf):
    c = conf.double()
    return float(100 * (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)).mean())


def sumar(conf, p, et):
    pred = F.interpolate(p.float(), et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
    m = et != IGNORAR
    conf += torch.bincount(40 * et[m] + pred[m], minlength=1600).view(40, 40)


def lotes(datos, lote=4):
    for k in range(0, len(datos["rgb"]), lote):
        rgb, prof, et = lote_a_gpu(datos, np.arange(k, min(k + lote, len(datos["rgb"]))))
        yield normalizar_rgb(rgb), prof / 1e3, prof > 0, et, prof


def profundidad_e7(test):
    out = []
    for s in range(3):
        modelo, _ = cargar(corrida("E7", s))
        modelo.eval()
        m = {k: [] for k in ("si", "absrel", "d1", "si_cte", "absrel_cte", "d1_cte")}
        for x, prof_m, valida, _, prof in lotes(test):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = modelo(x, prof_m, valida, todo=True)["prof"].float()
            tam = pred.shape[-2:]
            v = (prof > 0).float()
            num, den = F.adaptive_avg_pool2d(prof * v, tam), F.adaptive_avg_pool2d(v, tam)
            ok = den > 0.5
            lt_todo = torch.log((num / den.clamp_min(1e-6)).clamp_min(1.0) / 1e3)
            for i in range(len(pred)):
                lt = lt_todo[i][ok[i]]
                for p, suf in ((pred[i][ok[i]], ""), (torch.zeros_like(lt), "_cte")):
                    e = (p - lt) - (p - lt).mean()                     # escala alineada por imagen
                    r = torch.exp(e)
                    m["si" + suf].append(float(e.pow(2).mean().sqrt()))
                    m["absrel" + suf].append(float((r - 1).abs().mean()))
                    m["d1" + suf].append(float((torch.maximum(r, 1 / r) < 1.25).float().mean()))
        out.append({k: float(np.mean(v)) for k, v in m.items()})
        print(f"E7 s{s}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[-1].items()), flush=True)
        del modelo
        torch.cuda.empty_cache()
    return out


def descomponer_e8(test):
    out = []
    for s in range(3):
        modelo, _ = cargar(corrida("E8", s))
        modelo.eval()
        conf = {k: torch.zeros(40, 40, dtype=torch.long, device=DEV) for k in ("mezcla", "fusion", "color")}
        alfas = []
        for x, prof_m, valida, et, _ in lotes(test):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                sal = modelo(x, prof_m, valida, todo=True)
            sumar(conf["mezcla"], sal["seg"], et)
            sumar(conf["fusion"], sal["fus"], et)
            sumar(conf["color"], sal["rgb"], et)
            alfas.append(float(sal["alfa"].float().mean()))
        out.append({**{k: miou(c) for k, c in conf.items()}, "alfa_medio": float(np.mean(alfas))})
        print(f"E8 s{s}: " + "  ".join(f"{k} {v:.3f}" for k, v in out[-1].items()), flush=True)
        del modelo
        torch.cuda.empty_cache()
    return out


def probabilidades(exp, s, test):
    """Probabilidades del test con profundidad completa, en la CPU y en float16, para combinarlas después."""
    modelo, _ = cargar(corrida(exp, s))
    modelo.eval()
    p = []
    for x, prof_m, valida, _, _ in lotes(test):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p.append(modelo(x, prof_m, valida).float().softmax(1).half().cpu())
    del modelo
    torch.cuda.empty_cache()
    return torch.cat(p)


def promedios(test):
    P = {(e, s): probabilidades(e, s, test) for e in ("E1", "E1u", "E2") for s in range(3)}
    n = len(test["rgb"])

    def evaluar(p):
        conf = torch.zeros(40, 40, dtype=torch.long, device=DEV)
        for k in range(0, n, 4):
            _, _, et = lote_a_gpu(test, np.arange(k, min(k + 4, n)))
            sumar(conf, p[k:k + len(et)].to(DEV), et)
        return miou(conf)

    out = {f"{e} solo": [evaluar(P[(e, s)]) for s in range(3)] for e in ("E1", "E1u", "E2")}
    for e in ("E1", "E1u"):
        out[f"{e} + {e} (semillas distintas)"] = [evaluar((P[(e, a)].float() + P[(e, b)].float()) / 2)
                                                 for a, b in PARES]
    for a in ALFAS:                      # α es el peso del segundo modelo
        for e1, e2 in (("E1u", "E1"), ("E1", "E2"), ("E1u", "E2")):
            out[f"{e1} + {e2}, α = {a:g}"] = [evaluar((1 - a) * P[(e1, s)].float() + a * P[(e2, s)].float())
                                              for s in range(3)]
    for k, v in out.items():
        print(f"{k:32s} {st.mean(v):.2f} ± {st.stdev(v):.2f}", flush=True)
    return out


def por_clase():
    """IoU por clase con profundidad completa, media de las 3 semillas, y correlación entre las ganancias."""
    iou, clases = {}, None
    for exp in ("E1", "E1u", "E2", "E8", "E7"):
        rs = [json.loads((corrida(exp, s) / "resultado.json").read_text(encoding="utf-8")) for s in range(3)]
        iou[exp] = np.mean([r["resultados"]["completa"]["iou_clase"] for r in rs], 0)
        clases = rs[0]["clases"]
    prof, ajuste = iou["E2"] - iou["E1"], iou["E1u"] - iou["E1"]
    r = float(np.corrcoef(prof, ajuste)[0, 1])
    print(f"correlación por clase entre E2 − E1 y E1u − E1: {r:.2f}", flush=True)
    return {"clases": clases, "correlacion_E2-E1_E1u-E1": r,
            **{f"iou_{e}": [round(float(x), 2) for x in v] for e, v in iou.items()}}


def main():
    test = nyuv2.cache("test")
    out = {"nota": "exploratorio, sobre el test; no sirve para elegir modelos (D-04)", "por_clase": por_clase(),
           "profundidad_e7": profundidad_e7(test), "e8": descomponer_e8(test), "promedios": promedios(test)}
    (RUNS / "diagnosticos_d03.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
