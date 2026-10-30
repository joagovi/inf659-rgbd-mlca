# -*- coding: utf-8 -*-
"""Coste computacional de la red completa, de extremo a extremo, para E1-E5 y las variantes de D-06.

    python src/coste.py                          # en la GPU del equipo: bf16, lote 1, 480×640
    RGBD_SIN_PESOS=1 python src/coste.py --dispositivo cpu --precision fp32   # lo que se pudo medir al preparar D-06

Para cada experimento: parámetros totales y entrenables; FLOPs por pasada (contados por
`torch.utils.flop_counter.FlopCounterMode`, que cuenta 2 FLOPs por multiplicación-suma en convoluciones y
productos de matrices; MACs ≈ FLOPs / 2; no cuenta operaciones elemento a elemento, normalizaciones ni
activaciones), desglosados en codificador de color, DeFM, fusión y cabeza; latencia de extremo a extremo
(preprocesado de la profundidad incluido, sin la lectura del disco) con media y desviación sobre `--repeticiones`
pasadas tras `--calentamiento`; y memoria pico de la GPU. Los pesos no cambian el coste, así que con
RGBD_SIN_PESOS=1 las cifras de FLOPs y parámetros son las mismas; la latencia en CPU no representa la de la GPU.

Resultado: runs/coste/coste_<dispositivo>_<precisión>.json y .md. No usa NYUv2.
Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver `DECISIONES.md`.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import procedencia  # noqa: E402
import entrenar_rgbd as E  # noqa: E402

EXPS = ("E1", "E2", "E4", "E5", "E2c", "E3c", "E4c", "E3cd", "E4cd", "E1u", "E2u")
GRUPOS = {"color": "evit.backbone", "defm": "defm", "fusion": "fus", "cabeza": "evit.head"}


def contar(modelo):
    tot = sum(p.numel() for p in modelo.parameters())
    ent = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    por = {g: sum(p.numel() for n, p in modelo.named_parameters() if n.startswith(pref)) for g, pref in GRUPOS.items()}
    return tot, ent, por


def flops(modelo, x, prof, valida):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        modelo(x, prof, valida)
    cuentas = fc.get_flop_counts()
    total = sum(cuentas["Global"].values())
    por = {}
    for g, pref in GRUPOS.items():         # los módulos de cada parte que se llamaron, sin contar dos veces
        raiz = f"RGBD.{pref}"
        claves = [k for k in cuentas if k == raiz or k.startswith(raiz + ".")]
        claves = [k for k in claves if not any(k.startswith(o + ".") for o in claves if o != k)]
        por[g] = sum(sum(cuentas[k].values()) for k in claves)
    return total, por


def latencia(modelo, x, prof, valida, dispositivo, precision, calentamiento, repeticiones):
    amp = precision == "bf16"
    tiempos = []
    if dispositivo.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(calentamiento + repeticiones):
        if dispositivo.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast(dispositivo.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            modelo(x, prof, valida)
        if dispositivo.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= calentamiento:
            tiempos.append(1e3 * (time.perf_counter() - t))
    pico = torch.cuda.max_memory_allocated() / 2**20 if dispositivo.startswith("cuda") else None
    return st.mean(tiempos), st.stdev(tiempos) if len(tiempos) > 1 else 0.0, pico


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispositivo", default=E.DEV)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--lote", type=int, default=1)
    ap.add_argument("--calentamiento", type=int, default=5)
    ap.add_argument("--repeticiones", type=int, default=30)
    ap.add_argument("--exps", nargs="+", default=list(EXPS))
    a = ap.parse_args()
    dev = a.dispositivo
    E.DEV = dev
    torch.manual_seed(0)
    x = torch.randn(a.lote, 3, a.alto, a.ancho, device=dev)
    prof = 1 + 4 * torch.rand(a.lote, 1, a.alto, a.ancho, device=dev)
    valida = torch.ones_like(prof, dtype=torch.bool)
    filas = {}
    for exp in a.exps:
        modelo = E.construir(exp).to(dev).eval()
        tot, ent, por_p = contar(modelo)
        fl, por_f = flops(modelo, x, prof, valida)
        lat, de, pico = latencia(modelo, x, prof, valida, dev, a.precision, a.calentamiento, a.repeticiones)
        filas[exp] = {"parametros": tot, "entrenables": ent, "parametros_por_parte": por_p, "flops": fl,
                      "flops_por_parte": por_f, "latencia_ms": lat, "latencia_de_ms": de, "vram_pico_mib": pico}
        print(f"{exp}: {tot / 1e6:.2f} M ({ent / 1e6:.3f} M entrenables), {fl / 1e9:.2f} GFLOPs, "
              f"{lat:.1f} ± {de:.1f} ms" + (f", {pico:.0f} MiB" if pico else ""), flush=True)
        del modelo
    out = {"dispositivo": dev, "precision": a.precision, "resolucion": [a.alto, a.ancho], "lote": a.lote,
           "repeticiones": a.repeticiones, "calentamiento": a.calentamiento, "entorno": procedencia.entorno(),
           "hilos_cpu": torch.get_num_threads(), "experimentos": filas,
           "nota": "FLOPs de FlopCounterMode (conv y matmul; 2 por MAC). Latencia de extremo a extremo con la "
                   "profundidad ya en memoria; incluye la codificación de DeFM y la máscara."}
    sal = E.sg.RAIZ / "runs" / "coste"
    sal.mkdir(parents=True, exist_ok=True)
    nombre = f"coste_{dev.split(':')[0]}_{a.precision}"
    (sal / f"{nombre}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (sal / f"{nombre}.md").write_text(informe(out), encoding="utf-8")
    print(informe(out))


def informe(o):
    gpu = o["entorno"].get("gpu") or f"CPU ({o['hilos_cpu']} hilos, {o['entorno']['plataforma']})"
    L = [f"# Coste por experimento · {gpu} · {o['precision']} · lote {o['lote']} · "
         f"{o['resolucion'][0]}×{o['resolucion'][1]}", "",
         f"torch {o['entorno']['torch']}; pesos {'aleatorios (no cambian el coste)' if o['entorno']['sin_pesos'] else 'oficiales'}.",
         "", "| exp | parámetros (M) | entrenables (M) | GFLOPs total | color · DeFM · fusión · cabeza (GFLOPs) | "
         "latencia ms (media ± DE) | VRAM pico MiB |", "|---|---|---|---|---|---|---|"]
    for e, r in o["experimentos"].items():
        pf = r["flops_por_parte"]
        vram = "—" if r["vram_pico_mib"] is None else f"{r['vram_pico_mib']:.0f}"
        L.append(f"| {e} | {r['parametros'] / 1e6:.2f} | {r['entrenables'] / 1e6:.3f} | {r['flops'] / 1e9:.2f} | "
                 + " · ".join(f"{pf[g] / 1e9:.2f}" for g in GRUPOS) + f" | {r['latencia_ms']:.1f} ± "
                 f"{r['latencia_de_ms']:.1f} | {vram} |")
    L += ["", o["nota"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
