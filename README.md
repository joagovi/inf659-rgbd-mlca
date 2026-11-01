# Masked Linear Cross-Attention for Robust RGB-D Semantic Segmentation

**MLCA** fuses a frozen EfficientViT-B1 (RGB) and a frozen DeFM ResNet-18 (depth) with a linear cross-attention
that **excludes the pixels where the depth sensor gave no reading**. RGB features query; depth features answer.

$$o_i = \frac{\varphi(q_i)^\top S}{\varphi(q_i)^\top z + \varepsilon},\qquad
S = \sum_j m_j\,\varphi(k_j)\,v_j^\top,\qquad z = \sum_j m_j\,\varphi(k_j),\qquad \varphi = \mathrm{ReLU}$$

The validity mask $m_j = 0$ at depth holes removes them from both the summary and the normalizer, at linear
cost. The output projection has no bias and starts at zero, so with no valid depth MLCA contributes exactly
zero and the decoder receives only the RGB features.

Final project of *Deep Learning INF659* (MSc in Computer Science, Pontificia Universidad Católica del Perú,
2026-2) and poster at WAIMLAp 2026. **Work in progress.**

**Team:** Mónica Abarca · Marshall Fernández · Joaquin Gonzalez Villarreal · Claudia Suárez.

![Architecture](proposal/figs/arquitectura_mlca_en.png)

## Preliminary results (NYUv2, 40 classes, test, 3 seeds)

Frozen extractors; only the fusion and the EfficientViT head are trained. Predictions were written before
training ([`DECISIONES.md`](DECISIONES.md)).

| Depth condition | E1 · RGB only | E2 · RGB + depth, no mask |
|---|---|---|
| Full | 44.99 ± 0.30 | 47.31 ± 0.26 |
| Real sensor holes (`rawDepths`) | 44.99 ± 0.30 | 46.97 ± 0.52 |
| 50 % random holes | 44.99 ± 0.30 | 43.90 ± 0.12 |
| No depth (empty map) | 44.99 ± 0.30 | 42.25 ± 0.45 |

Adding depth without a mask helps while depth is available, but falls below RGB only when much of it is
missing: trained always with depth, E2 cannot tell a missing value from a measured one. E1 does not use depth,
so its value is the same in every condition. Details and all conditions:
[`notebooks/11_e1_e2_nyuv2.ipynb`](notebooks/11_e1_e2_nyuv2.ipynb).

![E1 and E2 per depth condition](proposal/figs/resultados_e1_e2_en.png)

| Experiment | Configuration | Status |
|---|---|---|
| E1 | EfficientViT, RGB only | done |
| E2 | EfficientViT + DeFM, residual 1×1 projection (concatenation with the RGB block fixed to identity) | done (exploratory) |
| E4, E5 | MLCA (fractional mask), with and without modality dropout | done, **not usable to test the hypothesis** (see below) |
| E2c, E3c, E4c, E3cd, E4cd | Corrected depth features; 2×2 factorial mask × modality dropout (binary mask, as in the abstract) | done (D-06; results below) |
| E4cf, E4cfd | Fractional-mask ablation | done (D-06) |
| E2cd | E2c with modality dropout | done (D-07) |
| E10, E11, E11d | Last RGB stage fine-tuned; E11 also adds depth at the input of that stage (sum); E11d adds modality dropout | done (D-05, D-07) |
| E3e, E4e, E4ed | MLCA instead of the sum at the input of the last RGB stage, without and with the mask; E4ed adds modality dropout | done (D-07) |
| E6 | E4 with a masked softmax attention (as in UCTNet) | future work |

**Audit D-06 (2026-10-29).** In the released DeFM ResNet-18 weights only the BiFPN P4 output depends on the input;
P3 and P5 are constant. With constant keys and values, linear attention returns the same value at every
position, so in E4 MLCA added a learned constant bias at 1/8 and 1/32 that the mask removes only when *no* depth
is valid: that is the collapse of E4 without depth (0.78 mIoU), not a property of masking. Also, without modality
dropout the mask never acts during training (filled depth), and the v3 mask was fractional while the abstract
specifies a binary one. The corrected design (family «c») uses depth features that depend on the input at all
three scales and isolates the mask as the only difference between E3c and E4c. All test-set numbers so far are
exploratory (the test set was used for diagnostics). Details: [`auditoria/`](auditoria/).

| E4 / E5 (v3, exploratory) | Full | Raw holes | Regions | 50 % random | No depth |
|---|---|---|---|---|---|
| E4 · MLCA | 45.24 ± 0.29 | 45.04 ± 0.19 | 44.87 ± 0.39 | 41.89 ± 1.40 | 0.78 ± 1.22 |
| E5 · MLCA + dropout 0.3 | 45.48 ± 0.21 | 45.45 ± 0.20 | 45.39 ± 0.07 | 44.25 ± 0.73 | 45.74 ± 0.10 |

## Results of the corrected design (D-06, D-07)

Test, mean of 3 seeds, mIoU (%). Each cell goes from *no dropout* to *modality dropout* (0.3); MLCA uses the
binary mask. In the encoder rows, depth also enters at the input of the last RGB stage, which is fine-tuned.

| Where depth enters | Fusion | Full depth | Raw sensor holes | No depth |
|---|---|---|---|---|
| head (frozen encoders) | sum (E2c → E2cd) | 47.53 → 48.05 | 43.32 → 47.66 | 20.33 → 45.47 |
| head (frozen encoders) | MLCA (E4c → E4cd) | 44.99 → 45.21 | 40.56 → 44.91 | 14.86 → 46.16 |
| encoder (last stage fine-tuned) | sum (E11 → E11d) | 51.06 → 51.26 | 47.71 → 50.41 | 15.75 → 47.43 |
| encoder (last stage fine-tuned) | MLCA (E4e → E4ed) | 49.39 → 49.24 | 47.78 → 48.49 | 38.75 → 47.29 |
| — | RGB only (E1 · E1u) | 44.99 · 47.31 | same | same |

![mIoU under depth degradation](proposal/figs/robustez_d06_d07_en.png)

- **With full depth, the sum beats MLCA in both places**: +2.5 to +2.8 in the head and +1.7 to +2.0 in the
  encoder. Depth helps more inside the encoder (E11 − E10 = +1.8).
- **Without dropout, MLCA in the encoder trades accuracy for robustness**: −1.7 with full depth, but +6.0 with 50 %
  random holes, +4.2 with 50 cm noise and +23.0 with no depth, mostly from the attention itself rather than the
  mask. In the head it loses in every condition.
- **Modality dropout gives robustness at no cost with full depth** (−0.15 to +0.52 in the four variants). With it,
  the sum stays ahead in every condition except no depth (a tie in the encoder; MLCA +0.7 in the head).
- **The binary mask acts only with large holes or no depth**: +9.4 with no depth in the encoder without dropout;
  −0.6 to +0.1 with the real sensor holes.
- **The best single model, E11d**, is 3.95 above RGB only with full depth and 3.10 with the sensor's holes, and
  matches it with no depth. DeFM takes 24.3 of the ~31.8 GFLOPs; MLCA adds 0.14–0.49 GFLOPs over the sum.

Pre-registered criterion (paired image bootstrap with Holm, same sign in the 3 seeds), all tables, mechanism and
cost: [`notebooks/12_resultados.ipynb`](notebooks/12_resultados.ipynb) and [`DECISIONES.md`](DECISIONES.md).

## Repository

```
proposal/      project proposal (Spanish), WAIMLAp extended abstract and poster, figures
src/
  mlca.py              the MLCA block
  rgbd.py              EfficientViT-B1 + DeFM ResNet-18 + fusion; DeFM preprocessing; depth conditions
  nyuv2.py             NYUv2 reader (official .mat, raw and filled depth, 40 classes)
  entrenar_rgbd.py     training and evaluation protocol
  efficientvit_seg.py  EfficientViT-Seg loader with the official weights
notebooks/
  09_mlca_bloque       the block on synthetic data: exact RGB path, exact hole exclusion, linear cost
  10_nyuv2_huecos      the real holes of NYUv2 raw depth: how many, where, which classes, which scale
  11_e1_e2_nyuv2       results of E1 and E2
  12_resultados        results of D-06 and D-07: all tables, robustness figure, criterion, mechanism, cost
  procedencia.py       pinned DeFM commit, checkpoint SHA-256 and environment, stored in every result (D-06)
  estadistica.py       statistics from the result files: mean ± SD, t intervals, paired differences, image bootstrap
  coste.py             parameters, FLOPs, end-to-end latency and peak VRAM of the full network
  diagnosticos_d06.py  MLCA diagnostics on the scene-level validation split
tests/         104 automated tests (CPU with synthetic data and random weights; 2 GPU-only)
auditoria/     D-06 audit: findings, coherence matrix, v2/v3 comparison, DeFM report, tables, plan, verification
DECISIONES.md  decisions and predictions written before each experiment
predicciones/  the AI's predictions for D-06 and D-07, written before each run, and how they compared
tools/         verificar_defm.py, huecos_efectivos.py, reproducir_historicos.py, cola_d06.sh, cola_d07.sh,
               criterio_d06.py (pre-registered criterion), run_notebook.py
```

Code and notebooks are documented in Spanish.

**Scripts produce the results; notebooks read and show them.** Training runs take 12–16 minutes each, so they
are launched from the terminal with `src/entrenar_rgbd.py`, which applies the same protocol to every experiment
(only `--exp` changes) and writes `runs/rgbd/<run>/resultado.json` with the mIoU per depth condition and per
class. `11_e1_e2_nyuv2` only reads those files. Notebooks 09 and 10 compute without training (1–3 minutes).

| File | Kind | Trains? | What it does |
|---|---|---|---|
| `src/entrenar_rgbd.py` | script | yes | Training, scene-level validation for the learning rate, final evaluation; `--reevaluar` adds new conditions to trained models |
| `notebooks/09_mlca_bloque` | notebook | no | Checks of the MLCA block on synthetic data; frozen encoders' feature shapes |
| `notebooks/10_nyuv2_huecos` | notebook | no | Real holes of NYUv2 raw depth |
| `notebooks/11_e1_e2_nyuv2` | notebook | no | Tables and figures of E1 and E2 from `resultado.json` |
| `notebooks/12_resultados` | notebook | no | Tables and figures of D-06 and D-07 from `resultado.json`, the criterion, the diagnostics and the cost (seconds) |

## Reproduce

```bash
git clone https://github.com/joagovi/inf659-rgbd-mlca.git && cd inf659-rgbd-mlca
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/mit-han-lab/efficientvit.git third_party/efficientvit
git -C third_party/efficientvit checkout de7d773
```

Data (not versioned), in `datos/nyuv2/`:

- `nyu_depth_v2_labeled.mat`: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- `splits.mat`: http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
- `classMapping40.mat`: https://github.com/ankurhanda/nyuv2-meta-data

DeFM code is loaded from `torch.hub` at a pinned commit (`leggedrobotics/defm:4849a36`) and both checkpoints from
Hugging Face; their SHA-256 is stored in every `resultado.json` (`src/procedencia.py`). Then:

```bash
pytest -q                                                   # tests; no data or weights needed (D-06)
python src/entrenar_rgbd.py --humo                          # smoke test: a few steps
python src/entrenar_rgbd.py --barrido E2                    # learning rate on a scene-level validation split
python src/entrenar_rgbd.py --exp E2 --lr 3e-4 --semilla 0  # train on the 795 images, evaluate on the 654
python tools/run_notebook.py notebooks/11_e1_e2_nyuv2.ipynb
```

One run takes 12–16 minutes on an RTX 4060 Laptop GPU (8 GB).

## Key references

- H. Cai et al., *EfficientViT: Multi-scale linear attention for high-resolution dense prediction*, ICCV 2023.
- M. Patel et al., *DeFM: Learning foundation representations from depth for robotics*, IEEE T-RO (conditionally
  accepted), 2026. arXiv:2601.18923.
- X. Ying and M. C. Chuah, *UCTNet: Uncertainty-aware cross-modal transformer network for indoor RGB-D semantic
  segmentation*, ECCV 2022 (closest prior work: masks invalid depth inside a softmax self-attention).
- C. Liao et al., *Benchmarking multi-modal semantic segmentation under sensor failures*, CVPR Workshops 2025.
- N. Silberman et al., *Indoor segmentation and support inference from RGBD images*, ECCV 2012 (NYUv2).

## AI assistance

Code and much of the text were written with the help of Claude (Anthropic), as allowed by the course and
declared in [`DECISIONES.md`](DECISIONES.md). The team wrote its predictions before E1, E2, E4 and E5. D-03 to D-05
ran without registered predictions, at the team's request. For D-06 and D-07 the team wrote none; the AI's
predictions, written before each run ([`predicciones/`](predicciones/)), serve as the pre-registration. The AI also
drafted the analysis of D-06 and D-07, which the team reviews.
