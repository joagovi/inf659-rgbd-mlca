# -*- coding: utf-8 -*-
"""Verificación de D-05: ¿P3 y P5 de DeFM ResNet-18 son constantes? ¿Por el checkpoint o por la extracción?

    # en la máquina del equipo, con los pesos oficiales y NYUv2 (CPU basta; unos minutos):
    python tools/verificar_defm.py --nyuv2 --n 64
    # sin pesos ni datos (controles, lo que se ejecutó al preparar D-06):
    python tools/verificar_defm.py --aleatorio --sinteticos
    python tools/verificar_defm.py --degenerado --sinteticos

Qué mide, en cada salida (P3, P4, P5 de `dense_bifpn`; C3, C4, C5 del ResNet; y la ruta C3/P4/C5 de D-05):

1. Procedencia: commit del código de DeFM, ruta, tamaño y SHA-256 del checkpoint.
2. Normas de los parámetros de la BiFPN: proyecciones, convoluciones de salida por nivel, pesos de fusión.
3. Variabilidad con profundidades reales: varianza espacial (dentro de cada imagen), varianza entre imágenes (de
   la media espacial) y cambio relativo entre imágenes distintas ‖f(a) − f(b)‖ / ‖f(a)‖, todo relativo a la
   norma cuadrática media del rasgo.
4. Sensibilidad: cambio relativo al alterar la profundidad de la misma imagen (escala ×1.1, ruido de 5 cm, 30 %
   de huecos aleatorios, rectángulos, profundidad cruda y profundidad vacía).
5. Extracción: `rgbd.rasgos_defm(..., "p3p4p5")` es exactamente `defm(x)["dense_bifpn"]`; P4 de «c3p4c5» es
   exactamente el P4 oficial; C3 y C5 son el LayerNorm sin parámetros de las salidas del ResNet.
6. Codificación: la oficial (`convert_to_3channel_metric`, mínimo y máximo con los ceros de los huecos) frente a la
   de D-01 (solo píxeles válidos): diferencia del tercer canal y cambio relativo de cada rasgo.

Controles sin pesos: `--aleatorio` usa la inicialización aleatoria (P3 y P5 deben variar: la ruta de extracción
no los vuelve constantes); `--degenerado` pone a cero las convoluciones de salida de P3 y P5 y los pesos de
fusión de P5, como describe D-05 (deben salir constantes). Si con los pesos oficiales P3 y P5 salen constantes y
con `--aleatorio` no, la causa es el checkpoint.

Resultado: JSON y markdown en runs/defm/ (o donde diga --salida).
Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import procedencia as P  # noqa: E402
import rgbd  # noqa: E402

NIVELES = ("P3", "P4", "P5", "C3", "C4", "C5", "D05_C3", "D05_P4", "D05_C5")


def profundidades_sinteticas(n, h=480, w=640, semilla=0):
    """Escenas de interior de juguete (suelo, paredes, cajas) en metros, con huecos tipo Kinect en la cruda."""
    rng = np.random.default_rng(semilla)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rell, cruda = [], []
    for _ in range(n):
        fondo = 2.0 + 4.0 * rng.random()
        horizonte = h * (0.4 + 0.3 * rng.random())
        d = np.where(yy > horizonte, 1.0 + (fondo - 1.0) * (h - yy) / (h - horizonte + 1), fondo)
        d = d + 0.3 * rng.random() * (xx / w - 0.5)
        for _ in range(rng.integers(2, 6)):
            y0, x0 = rng.integers(0, h - 60), rng.integers(0, w - 60)
            d[y0:y0 + rng.integers(30, 200), x0:x0 + rng.integers(30, 250)] = 0.8 + 3 * rng.random()
        d = d.astype(np.float32)
        c = d.copy()
        c[:, :40] = 0
        c[:20] = 0
        for _ in range(rng.integers(1, 5)):
            y0, x0 = rng.integers(0, h - 40), rng.integers(0, w - 40)
            c[y0:y0 + rng.integers(5, 60), x0:x0 + rng.integers(5, 60)] = 0
        rell.append(d)
        cruda.append(c)
    return rell, cruda


def profundidades_nyuv2(n, semilla=0):
    import nyuv2
    datos = nyuv2.cache("test")
    idx = np.sort(np.random.default_rng(semilla).choice(len(datos["prof"]), n, replace=False))
    return ([datos["prof"][i].astype(np.float32) / 1e3 for i in idx],
            [datos["cruda"][i].astype(np.float32) / 1e3 for i in idx], idx.tolist())


def oficial_3canales(prof_m):
    """`convert_to_3channel_metric` de DeFM (defm/utils/utils.py, commit fijado), en numpy, + media y desviación."""
    d = np.clip(np.nan_to_num(prof_m, nan=0.0, posinf=0.0, neginf=0.0), 0, 100.0)
    lg = np.log1p(d)
    c1, c2 = lg / np.log1p(100.0), np.clip(lg / np.log1p(9.0), 0, 1)
    lo, hi = np.log1p(d.min()), np.log1p(d.max())
    c3 = (lg - lo) / (hi - lo) if hi > lo else np.zeros_like(lg)
    x = torch.from_numpy(np.stack([c1, c2, c3])[None].astype(np.float32))
    return (x - rgbd.DEFM_MEAN) / rgbd.DEFM_STD


@torch.no_grad()
def todas(defm, prof_m, valida=None, entrada=None):
    """Todas las salidas para una profundidad (1,1,H,W) en metros."""
    if valida is None:
        valida = prof_m > 0
    x = entrada if entrada is not None else rgbd.entrada_defm(prof_m, valida)
    o = defm(x)["dense_bifpn"]
    c = defm.backbone(x)
    d05 = rgbd.rasgos_defm(defm, prof_m, valida, "c3p4c5") if entrada is None else {}
    out = {"P3": o["P3"], "P4": o["P4"], "P5": o["P5"], "C3": c["c3"], "C4": c["c4"], "C5": c["c5"]}
    out.update({f"D05_{k}": v for k, v in d05.items()})
    return out


def rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def normas(defm):
    out = {}
    for k, p in defm.named_parameters():
        if k.startswith("fpn.") or k.startswith("norm_patch"):
            out[k] = {"norma": float(p.norm()), "forma": list(p.shape)}
            if p.numel() <= 4:
                out[k]["valores"] = [float(x) for x in p.flatten()]
    return out


def variabilidad(feats):
    """feats: lista de (1,C,h,w) para imágenes distintas."""
    f = torch.cat(feats)                                     # (N,C,h,w)
    rms = float(f.pow(2).mean().sqrt())
    esp = float(f.var(dim=(2, 3)).mean().sqrt())             # desviación espacial media
    medias = f.mean(dim=(2, 3))                              # (N,C)
    entre = float(medias.var(dim=0).mean().sqrt())           # desviación entre imágenes de la media espacial
    pares = [rel(f[i], f[(i + 1) % len(f)]) for i in range(len(f))]
    return {"rms": rms, "desv_espacial_rel": esp / max(rms, 1e-30), "desv_entre_imagenes_rel": entre / max(rms, 1e-30),
            "cambio_entre_imagenes_rel": float(np.mean(pares)), "cambio_entre_imagenes_min": float(np.min(pares))}


def perturbaciones(rell, cruda, semilla=0):
    rng = np.random.default_rng(semilla)
    out = {"escala_1.1": lambda d, c: d * 1.1, "ruido_5cm": lambda d, c: np.clip(
        d + rng.normal(0, 0.05, d.shape), 0.1, None).astype(np.float32),
        "huecos_30": lambda d, c: np.where(rng.random(d.shape) < 0.3, 0, d).astype(np.float32),
        "rectangulos": lambda d, c: np.where(rgbd.huecos_regiones(*d.shape, 0.5, rng), 0, d).astype(np.float32),
        "cruda": lambda d, c: c, "vacia": lambda d, c: np.zeros_like(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--aleatorio", action="store_true", help="pesos aleatorios (control)")
    g.add_argument("--degenerado", action="store_true", help="aleatorios con P3 y P5 anulados como en D-05")
    d = ap.add_mutually_exclusive_group()
    d.add_argument("--nyuv2", action="store_true")
    d.add_argument("--sinteticos", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--salida", default=str(RAIZ / "runs" / "defm"))
    a = ap.parse_args()
    torch.manual_seed(0)
    modo = "aleatorio" if a.aleatorio else "degenerado" if a.degenerado else "oficial"
    defm = rgbd.cargar_defm(pesos=(modo == "oficial")).eval()
    if a.degenerado:                                   # lo que D-05 describe en el checkpoint oficial
        with torch.no_grad():
            for i in (0, 2):
                defm.fpn.blocks[0].convs[i].weight.mul_(1e-9 / defm.fpn.blocks[0].convs[i].weight.norm())
            defm.fpn.blocks[0].fuse_bottomup[1].weights.zero_()
    if a.nyuv2:
        rell, cruda, idx = profundidades_nyuv2(a.n)
        fuente = f"NYUv2 test, {a.n} imágenes {idx[:5]}..."
    else:
        rell, cruda = profundidades_sinteticas(a.n)
        fuente = f"{a.n} profundidades sintéticas (480×640)"
    T = lambda z: torch.from_numpy(z)[None, None]   # noqa: E731

    res = {"modo": modo, "fuente": fuente, "procedencia": P.entorno(), "normas_bifpn": normas(defm)}
    base = [todas(defm, T(z)) for z in rell]
    res["variabilidad"] = {k: variabilidad([b[k] for b in base]) for k in NIVELES}
    sens = {}
    for nombre, fn in perturbaciones(rell, cruda).items():
        alt = [todas(defm, T(fn(z, c))) for z, c in zip(rell, cruda)]
        sens[nombre] = {k: float(np.mean([rel(b[k], x[k]) for b, x in zip(base, alt)])) for k in NIVELES}
    res["sensibilidad_rel"] = sens
    # 5. extracción
    x = rgbd.entrada_defm(T(rell[0]), T(rell[0]) > 0)
    with torch.no_grad():
        o = defm(x)["dense_bifpn"]
        p345 = rgbd.rasgos_defm(defm, T(rell[0]), T(rell[0]) > 0, "p3p4p5")
        c345 = rgbd.rasgos_defm(defm, T(rell[0]), T(rell[0]) > 0, "c3p4c5")
        bb = defm.backbone(x)
    ln = lambda t: F.layer_norm(t.permute(0, 2, 3, 1), t.shape[1:2]).permute(0, 3, 1, 2)   # noqa: E731
    res["extraccion"] = {
        "p3p4p5_igual_oficial": all(torch.equal(p345[k], o[k]) for k in o),
        "d05_P4_igual_oficial_maxdif": float((c345["P4"] - o["P4"]).abs().max()),
        "d05_C3_igual_LN_c3_maxdif": float((c345["C3"] - ln(bb["c3"])).abs().max()),
        "d05_C5_igual_LN_c5_maxdif": float((c345["C5"] - ln(bb["c5"])).abs().max())}
    # 6. codificación oficial frente a D-01, con la cruda (huecos)
    cod = {"tercer_canal_dif_mediana_validos": [], **{k: [] for k in ("P3", "P4", "P5", "C3", "C4", "C5")}}
    for c in cruda:
        v = c > 0
        if v.all() or not v.any():
            continue
        xo, xn = oficial_3canales(c), rgbd.entrada_defm(T(c), T(c) > 0)
        canal = (xo[0, 2] - xn[0, 2]) * rgbd.DEFM_STD[0, 2, 0, 0]
        cod["tercer_canal_dif_mediana_validos"].append(float(canal[torch.from_numpy(v)].abs().median()))
        fo, fn_ = todas(defm, T(c), entrada=xo), todas(defm, T(c), entrada=xn)
        for k in ("P3", "P4", "P5", "C3", "C4", "C5"):
            cod[k].append(rel(fn_[k], fo[k]))
    res["codificacion_oficial_vs_d01"] = {k: (float(np.mean(v)) if v else None) for k, v in cod.items()}
    res["entrada_vacia_igual_oficial"] = bool(torch.allclose(
        oficial_3canales(np.zeros_like(rell[0])), rgbd.entrada_defm(T(np.zeros_like(rell[0])),
                                                                      T(np.zeros_like(rell[0])) > 0)))
    res["entrada_completa_igual_oficial_maxdif"] = float((oficial_3canales(rell[0]) - rgbd.entrada_defm(
        T(rell[0]), T(rell[0]) > 0)).abs().max())

    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    nombre = f"verificacion_defm_{modo}_{'nyuv2' if a.nyuv2 else 'sinteticos'}"
    (sal / f"{nombre}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (sal / f"{nombre}.md").write_text(informe(res), encoding="utf-8")
    print(informe(res))


def informe(r):
    L = [f"# Verificación de DeFM ResNet-18 · modo {r['modo']}", "", f"Fuente de profundidad: {r['fuente']}.", "",
         f"- Commit del código de DeFM: `{r['procedencia']['defm_commit']}`",
         f"- Checkpoint: `{json.dumps(r['procedencia']['pesos'].get('defm_resnet18.pth', 'sin pesos (control)'))}`",
         f"- torch {r['procedencia']['torch']}, {r['procedencia']['plataforma']}", "", "## Normas de la BiFPN", "",
         "| parámetro | forma | norma | valores |", "|---|---|---|---|"]
    for k, v in r["normas_bifpn"].items():
        if "convs" in k or "fuse" in k or "proj" in k:
            L.append(f"| `{k}` | {v['forma']} | {v['norma']:.3g} | {v.get('valores', '')} |")
    L += ["", "## Variabilidad entre imágenes y dentro de cada imagen (relativa al RMS del rasgo)", "",
          "| salida | RMS | desv. espacial | desv. entre imágenes | cambio entre imágenes (media / mín) |",
          "|---|---|---|---|---|"]
    for k, v in r["variabilidad"].items():
        L.append(f"| {k} | {v['rms']:.3g} | {v['desv_espacial_rel']:.3g} | {v['desv_entre_imagenes_rel']:.3g} | "
                 f"{v['cambio_entre_imagenes_rel']:.3g} / {v['cambio_entre_imagenes_min']:.3g} |")
    L += ["", "## Sensibilidad a cambios de la profundidad de la misma imagen (cambio relativo medio)", "",
          "| perturbación | " + " | ".join(NIVELES) + " |", "|---" * (len(NIVELES) + 1) + "|"]
    for p, v in r["sensibilidad_rel"].items():
        L.append(f"| {p} | " + " | ".join(f"{v[k]:.3g}" for k in NIVELES) + " |")
    L += ["", "## Extracción", "", *[f"- {k}: {v}" for k, v in r["extraccion"].items()], "",
          "## Codificación oficial frente a D-01 (profundidad con huecos)", "",
          *[f"- {k}: {v if v is None else f'{v:.3g}'}" for k, v in r["codificacion_oficial_vs_d01"].items()],
          f"- entrada vacía igual a la oficial: {r['entrada_vacia_igual_oficial']}",
          f"- entrada completa, diferencia máxima con la oficial: {r['entrada_completa_igual_oficial_maxdif']:.3g}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
