# 4 · Informe de verificación de DeFM (D-05)

**Pregunta.** En DeFM ResNet-18, ¿P3 (1/8) y P5 (1/32) de `dense_bifpn` son constantes? ¿Se debe al checkpoint
o a la forma de extraerlas? ¿Las salidas C3/P4/C5 de D-05 dependen de la entrada?

**Respuesta corta.** Lo que se pudo comprobar aquí apoya D-05 y apunta al **checkpoint**. Falta lo que necesita
los pesos oficiales: las normas, el SHA-256 y la variabilidad con profundidades reales. Queda listo en
`tools/verificar_defm.py --nyuv2`; en una CPU tarda minutos.

## 1. Procedencia

| | Valor |
|---|---|
| Código de DeFM | `leggedrobotics/defm`, commit **`4849a368d669ae5d34bcd612fa3620f0e8a7ab83`** (2026-10-26), clonado de GitHub para esta verificación |
| Cambios del código del modelo | `defm/models/` (ResNetBiFPN, BiFPN) **sin cambios entre `63ec5e1` (27/04/2026) y `4849a36`**: la red que construyeron las corridas de la v3 (21-29/10, `defm:main`) es esta |
| Cambio relevante en `main` | `b5538c8` (22/10): al descargar los pesos, la configuración de la arquitectura se lee de `config.json` de Hugging Face. Para resnet18, la configuración empaquetada es `arch: resnet18, out_channels: 128` |
| Checkpoint | `defm_resnet18.pth` de `huggingface.co/leggedrobotics/defm`. **No se pudo descargar** (proxy: 403). SHA-256 pendiente; se registra solo en cada corrida y se puede fijar en `procedencia.SHA256_ESPERADO` |

## 2. Lo que dice el código oficial

- `ResNetBiFPN.forward` devuelve `dense_bifpn = {P3, P4, P5}` = LayerNorm(conv₃ₓ₃(fusión BiFPN)) por nivel.
- El README oficial: `spatial_tokens = output["dense_bifpn"]["P4"]  # (B, C, H/16, W/16), the features distilled from the ViT`.
- Los *benchmarks* oficiales de segmentación y clasificación (`benchmarks/encoders.py`, `CNNEncoder`) usan **solo
  P4**. El cuaderno de inferencia también usa `forward_no_bifpn` (C3-C5 del ResNet sin normalizar).
- Lectura: la destilación supervisa P4 (y el token global). Nada empuja a P3 ni a P5 a ser útiles, y el
  decaimiento de pesos puede llevar sus convoluciones de salida a cero. Es coherente con las normas de ~1e-9 que
  midió D-05. Es una **hipótesis sobre el entrenamiento de DeFM**; no está confirmada por sus autores.

## 3. Controles ejecutados (sin pesos oficiales, profundidades sintéticas de 480×640, CPU)

`python tools/verificar_defm.py --aleatorio --sinteticos --n 8` y `--degenerado`. Resultados completos en `runs/defm/`.

| Salida | Pesos aleatorios: cambio entre imágenes | Pesos aleatorios: desv. espacial | Anulación de D-05: cambio entre imágenes | Anulación: sensibilidad a huecos 30 % |
|---|---|---|---|---|
| P3 | 0.62 | 0.56 | **8.3e-10** | 8.2e-10 |
| P4 | 0.64 | 0.60 | 0.64 | 0.77 |
| P5 | 0.62 | 0.60 | **0** | 0 |
| C3 (D-05, LN) | 0.85 | 0.76 | 0.85 | 1.05 |
| C5 (D-05, LN) | 0.72 | 0.66 | 0.72 | 0.83 |

(Cambio relativo ‖f(a) − f(b)‖/‖f(a)‖ y desviación espacial relativa al RMS. La «anulación de D-05» reproduce lo
que D-05 dice del checkpoint: `convs.0` y `convs.2` con norma 1e-9 y `fuse_bottomup.1` a cero.)

- **La extracción no produce la constancia:** con la misma ruta de código, P3 y P5 varían igual que P4 cuando los
  pesos no están anulados.
- **La anulación reproduce las cifras de D-05:** P3 cambia ~1e-9 (D-05: 4e-10) y P5 exactamente 0 (D-05: «nada»).
  Con `fuse_bottomup.1 = 0` la fusión de P5 vale 0, y su salida es solo el sesgo de la convolución, normalizado.
- **La extracción es la oficial:** `rasgos_defm(..., "p3p4p5")` es `defm(x)["dense_bifpn"]` bit a bit; el P4 de
  `c3p4c5` es el P4 oficial (diferencia 0); C3 y C5 son el LayerNorm sin parámetros de `backbone(x)` (diferencia 0).
  También lo cubre `tests/test_modelo.py::test_extraccion_de_rasgos_es_la_oficial`.
- **Codificación de la entrada:** sin huecos, la de D-01 coincide con la oficial (diferencia máxima 4.8e-7). Con la
  profundidad vacía, son idénticas. Con huecos difiere el tercer canal (mediana 0.054 en estas escenas sintéticas;
  D-01 midió 0.21 en NYUv2).

## 4. Evidencia con datos reales que ya existe (v3)

`runs/rgbd/diagnosticos_d04.json`, «permutadas»: E2 y E2u recibieron en una escala la profundidad de otra imagen
del test (una permutación sin puntos fijos).

| Modelo | propia | otra en 1/8 | otra en 1/16 | otra en 1/32 |
|---|---|---|---|---|
| E2 (3 semillas) | 47.54 / 47.04 / 47.36 | igual | 42.59 / 42.34 / 42.54 | igual |
| E2u (3 semillas) | 48.64 / 48.42 / 48.71 | igual | 40.65 / 39.76 / 41.00 | igual |

Que no cambie ni en el segundo decimal es lo esperable si P3 y P5 no dependen de la imagen.

## 5. Pendiente (con los pesos oficiales y NYUv2)

```bash
python tools/verificar_defm.py --nyuv2 --n 64        # runs/defm/verificacion_defm_oficial_nyuv2.{json,md}
```

Mide las normas de todas las capas de la BiFPN, el SHA-256 y el tamaño del checkpoint, la varianza espacial y
entre imágenes de P3/P4/P5/C3/C4/C5, la sensibilidad a escala, ruido, huecos, rectángulos, cruda y vacía, y la
diferencia entre la codificación oficial y la de D-01. **Criterio de decisión, escrito antes de ejecutarlo:**
D-05 se confirma si el cambio entre imágenes de P3 y P5 es < 1e-6 y el de P4, C3 y C5 > 1e-2. Si P3 o P5 superan
1e-2, el diagnóstico de D-05 es incorrecto y hay que revisar la ruta de extracción de la v3.

## 6. ¿Son mejores C3/P4/C5?

**No se ha probado.** Solo se sabe que dependen de la entrada (con pesos aleatorios; con los oficiales,
pendiente). C3 y C5 son rasgos del ResNet que la destilación no supervisó directamente. Que E2c supere a E2, o que
E4c no colapse, hay que medirlo (prioridades 1 y 4 del plan). La v4 no reemplaza la ruta de la v3: `p3p4p5`
sigue siendo la de E1-E11 y `c3p4c5` solo la usan E10, E11 y la familia «c».
