# -*- coding: utf-8 -*-
"""Diagnósticos de los resultados de D-05, sin reentrenar. Son exploratorios: nada de lo que aquí se mide sirve
para elegir un modelo.

1. Por clase (de los resultado.json, sin GPU): E10 − E2u (rasgos que dependen de la entrada), E11 − E10 (la
   profundidad también en el codificador) y E11 − E1u.
2. Qué usa la profundidad en E10 y E11, con profundidad completa en el test: como `diagnosticos_d04.permutadas`, la
   profundidad de una escala se cambia por la de otra imagen (la suma conserva su tamaño y pierde la información
   de la imagen); en E11, también la que entra en la última etapa del color. Además, el tamaño relativo de cada
   suma frente al rasgo de color, y de dónde viene la caída sin profundidad: la profundidad vacía («sin») solo en
   el codificador, solo en la cabeza o en los dos (lo último es la condición «sin»).
3. Diversidad frente a E1u: cuánto se alejan de ADE20K los pesos de la última etapa del color, y, en el test con
   profundidad completa, en qué fracción de los píxeles etiquetados coincide cada modelo con E1u de la misma
   semilla y cuánto da el promedio 0.5 / 0.5 de los dos. Con profundidad completa, la mezcla con tope 0.5 es ese
   promedio; el control es E1u con otra semilla. `combinar.py` empareja modelos de la misma semilla, que ven las
   imágenes en el mismo orden y con el mismo aumento; por eso se repite con el E1u de la semilla siguiente.

    python src/diagnosticos_d05.py

Resultado en runs/rgbd/diagnosticos_d05.json. Si ya existe, solo se calcula lo que le falte.

Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import efficientvit_seg as S  # noqa: E402
import nyuv2  # noqa: E402
import rgbd  # noqa: E402
from diagnosticos_d03 import miou, sumar  # noqa: E402
from entrenar_rgbd import DEV, IGNORAR, RUNS, cargar, lote_a_gpu, nombre_corrida, normalizar_rgb  # noqa: E402

LR = {"E1u": 1e-3, "E2u": 3e-3, "E10": 1e-3, "E11": 3e-3}    # elegidas en la validación (DECISIONES.md)
ESCALA = {"stage2": "1/8", "stage3": "1/16", "stage4": "1/32"}


def corrida(exp, s):
    return RUNS / nombre_corrida(exp, LR[exp], s)


def por_clase():
    iou, clases = {}, None
    for exp in LR:
        rs = [json.loads((corrida(exp, s) / "resultado.json").read_text(encoding="utf-8")) for s in range(3)]
        iou[exp] = np.mean([r["resultados"]["completa"]["iou_clase"] for r in rs], 0)
        clases = rs[0]["clases"]
    gan = {"E10-E2u": iou["E10"] - iou["E2u"], "E11-E10": iou["E11"] - iou["E10"], "E11-E1u": iou["E11"] - iou["E1u"],
           "E2u-E1u": iou["E2u"] - iou["E1u"]}
    corr = {"E11-E10_vs_E2u-E1u": float(np.corrcoef(gan["E11-E10"], gan["E2u-E1u"])[0, 1]),
            "E10-E2u_vs_E11-E10": float(np.corrcoef(gan["E10-E2u"], gan["E11-E10"])[0, 1])}
    print("correlaciones por clase: " + "  ".join(f"{k} {v:.2f}" for k, v in corr.items()), flush=True)
    return {"clases": clases, "correlaciones": corr,
            **{f"iou_{k}": [round(float(x), 2) for x in v] for k, v in iou.items()},
            **{f"ganancia_{k}": [round(float(x), 2) for x in v] for k, v in gan.items()}}


def que_usa(carpeta, datos, lote=4):
    """mIoU con profundidad completa cambiando, en cada variante, de dónde sale la profundidad de cada suma: la
    propia, la de otra imagen del test (una permutación fija sin puntos fijos) o la vacía de la condición «sin».
    También ‖W·f_prof‖ / ‖f_color‖ en cada escala y, en E11, ‖W'·P4‖ / ‖f_1/16‖ a la entrada de la última etapa."""
    modelo, _ = cargar(carpeta)
    modelo.eval()
    n = len(datos["rgb"])
    orden = np.random.default_rng(0).permutation(n)
    otra = np.empty(n, int)
    otra[orden] = np.roll(orden, -1)                      # un solo ciclo: ninguna imagen se queda con la suya
    cabeza = [c for c, _ in modelo.pares]
    variantes = {"propia": ("propia", {}),                # (codificador, {escala de la cabeza: fuente})
                 **{f"otra_{ESCALA[c]}": ("propia", {c: "otra"}) for c in cabeza},
                 "otra_cabeza": ("propia", dict.fromkeys(cabeza, "otra")),
                 "sin_cabeza": ("propia", dict.fromkeys(cabeza, "sin"))}
    if modelo.codificador:
        variantes |= {"otra_codificador": ("otra", {}), "otra_todas": ("otra", dict.fromkeys(cabeza, "otra")),
                      "sin_codificador": ("sin", {}), "sin_todas": ("sin", dict.fromkeys(cabeza, "sin"))}
    conf = {v: torch.zeros(40, 40, dtype=torch.long, device=DEV) for v in variantes}
    rel = {c: [] for c in cabeza + ["codificador"]}

    def rasgos(posiciones, cond):
        pv = [rgbd.condicion(cond, datos["prof"][i], datos["cruda"][i], semilla=int(i)) for i in posiciones]
        prof = torch.from_numpy(np.stack([p for p, _ in pv])).to(DEV)[:, None]
        valida = torch.from_numpy(np.stack([v for _, v in pv])).to(DEV)[:, None]
        return rgbd.rasgos_defm(modelo.defm, prof, valida, modelo.rasgos_prof)

    def norma(t):
        return t.float().flatten(1).norm(dim=1)

    for k in range(0, n, lote):
        idx = np.arange(k, min(k + lote, n))
        rgb, _, et = lote_a_gpu(datos, idx)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fd = {"propia": rasgos(idx, "completa"), "otra": rasgos(otra[idx], "completa"), "sin": rasgos(idx, "sin")}
            f = modelo.color(normalizar_rgb(rgb), 3 if modelo.codificador else 4)
            color = {}
            for q in {cod for cod, _ in variantes.values()}:
                color[q] = f
                if modelo.codificador:                    # como `RGBD.forward`
                    termino = modelo.fus_cod(fd[q]["P4"].to(f["stage3"].dtype))
                    x = modelo.etapa(4, f["stage3"] + termino)
                    color[q] = {**f, "stage4": x, "stage_final": x}
                    if q == "propia":
                        rel["codificador"] += (norma(termino) / norma(f["stage3"])).tolist()
            for v, (cod, cab) in variantes.items():
                fc = color[cod]
                g = {**fc, **{c: fc[c] + modelo.fus[c](fd[cab.get(c, "propia")][d].to(fc[c].dtype))
                              for c, d in modelo.pares}}
                sumar(conf[v], rgbd.salida(modelo.evit.head, g), et)
            fc = color["propia"]
            for c, d in modelo.pares:
                rel[c] += (norma(modelo.fus[c](fd["propia"][d].to(fc[c].dtype))) / norma(fc[c])).tolist()
    out = {**{v: miou(c) for v, c in conf.items()},
           **{f"relativo_{ESCALA.get(c, c)}": float(np.mean(r)) for c, r in rel.items() if r}}
    print(f"{carpeta.name}: " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    del modelo
    torch.cuda.empty_cache()
    return out


def cambio_etapa4():
    """‖θ − θ_ADE20K‖ / ‖θ_ADE20K‖ de los parámetros de la última etapa del color (sin las estadísticas de BN, que
    no se entrenan) de cada modelo, a partir de su cabeza.pt."""
    base = S.cargar_efficientvit("b1", "ade20k").backbone.stages[3]
    nombres = [k for k, _ in base.named_parameters()]
    theta0 = torch.cat([p.detach().flatten() for _, p in base.named_parameters()]).float()
    out = {}
    for exp in LR:
        out[exp] = []
        for s in range(3):
            estado = torch.load(corrida(exp, s) / "cabeza.pt", map_location="cpu")
            theta = torch.cat([estado[f"evit.backbone.stages.3.{k}"].flatten() for k in nombres]).float()
            out[exp].append(float((theta - theta0).norm() / theta0.norm()))
        print(f"{exp}: cambio relativo de la última etapa " + " / ".join(f"{x:.4f}" for x in out[exp]), flush=True)
    return out


def acuerdo(s, datos, ref=None, lote=4):
    """Con profundidad completa: coincidencia por píxel con E1u de la semilla `ref` (por defecto, s), mIoU de cada
    modelo de la semilla s (promediando probabilidades, como `combinar.py`) y del promedio 0.5 / 0.5 con ese E1u."""
    ref = s if ref is None else ref
    modelos = {"E1u": corrida("E1u", ref), "E1u_otra": corrida("E1u", (ref + 1) % 3),
               **{exp: corrida(exp, s) for exp in ("E2u", "E10", "E11")}}
    modelos = {k: cargar(c)[0].eval() for k, c in modelos.items()}
    conf = {f"{k}{v}": torch.zeros(40, 40, dtype=torch.long, device=DEV) for k in modelos for v in ("", "+E1u")}
    igual, total = dict.fromkeys(modelos, 0), 0
    n = len(datos["rgb"])
    for k in range(0, n, lote):
        idx = np.arange(k, min(k + lote, n))
        rgb, prof, et = lote_a_gpu(datos, idx)
        x = normalizar_rgb(rgb)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = {m: modelo(x, prof / 1e3, prof > 0).float().softmax(1) for m, modelo in modelos.items()}
        m = et != IGNORAR
        pred = {q: F.interpolate(v, et.shape[-2:], mode="bilinear", align_corners=False).argmax(1)[m]
                for q, v in p.items()}
        total += int(m.sum())
        for q in modelos:
            igual[q] += int((pred[q] == pred["E1u"]).sum())
            sumar(conf[q], p[q], et)
            sumar(conf[f"{q}+E1u"], (p[q] + p["E1u"]) / 2, et)
    out = {**{q: miou(c) for q, c in conf.items() if q != "E1u+E1u"},
           **{f"coincide_{q}": 100 * igual[q] / total for q in modelos if q != "E1u"}}
    print(f"semilla {s}, E1u {ref}: " + "  ".join(f"{k} {v:.2f}" for k, v in out.items()), flush=True)
    del modelos
    torch.cuda.empty_cache()
    return out


def main():
    test = nyuv2.cache("test")
    archivo = RUNS / "diagnosticos_d05.json"
    out = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {
        "nota": "exploratorio, sin reentrenar; no sirve para elegir modelos (D-05)"}
    pasos = {
        "por_clase": por_clase,
        "que_usa": lambda: {exp: [que_usa(corrida(exp, s), test) for s in range(3)] for exp in ("E10", "E11")},
        "cambio_etapa4": cambio_etapa4,
        "acuerdo": lambda: [acuerdo(s, test) for s in range(3)],
        "acuerdo_cruzado": lambda: [acuerdo(s, test, ref=(s + 1) % 3) for s in range(3)],
    }
    for clave, paso in pasos.items():
        if clave not in out:
            out[clave] = paso()
            archivo.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
