# 7 · Qué se verificó de verdad y qué queda pendiente

## Entorno donde se ejecutó todo lo de esta auditoría

Contenedor Linux sin GPU, con 2 vCPU y 7 GB de RAM. Python 3.12, torch 2.5.1 (la versión de `requirements.txt`),
torchvision 0.20.1, numpy 2.1.3 (`requirements.txt` fija 2.4.4), scipy 1.18.1, h5py 3.16.0 y pytest 9.2.0.
Código de DeFM: un clon de GitHub en `4849a36` (`RGBD_DEFM_REPO`). EfficientViT: `third_party/efficientvit` de la
v3 (`de7d773`). **Sin** los pesos oficiales (Hugging Face bloqueado por el proxy) y **sin** NYUv2 (servidor
bloqueado). Las redes se construyeron con su arquitectura real y pesos aleatorios fijos (`RGBD_SIN_PESOS=1`), y
los datos fueron sintéticos (`RGBD_DATOS_SINTETICOS=1`).

## Ejecutado

| Qué | Resultado | Evidencia |
|---|---|---|
| Suite de pruebas en CPU | **91 aprobadas, 2 saltadas (GPU), 0 fallidas**, en 3 min 12 s | `evidencia/pytest_cpu.txt`, `evidencia/pytest_cpu.xml` |
| · `test_mlca.py` (11) | Cero exacto sin profundidad; huecos excluidos aunque sean NaN; los tres modos iguales con m = 1; binaria ≠ fraccional con huecos; K/V constantes → sesgo; gradientes al inicio; bf16 estable; coste lineal | |
| · `test_condiciones.py` (21) | Las 13 condiciones son deterministas, están en metros y quitan la fracción nominal; la codificación de DeFM es igual a la oficial sin huecos; los datos sintéticos tienen la misma forma que la caché | |
| · `test_modelo.py` (22) | E1 ignora la profundidad; al inicio, E2/E4/E2c/E3c/E4c/E4cf dan los logits de E1; E3c, E4c y E4cf son la misma red; sin profundidad, E4 es la ruta de color y E3 no; solo se entrena lo previsto; la correspondencia de escalas y canales es correcta; la extracción es la oficial; P3/P5 constantes solo con los pesos anulados | |
| · `test_entrenamiento.py` (20) | Los 11 experimentos entrenan y evalúan, con huella y `por_imagen.npz` coherentes; E3c = E4c = E4cf bit a bit sin dropout, E4cd = E4cfd, E3cd ≠ E4cd; el dropout no cambia el orden de los datos (y en la v3 sí); la reanudación es bit a bit; ConfigConflict; `error.log`; `completo` + `derivar`; respaldo exacto de la mezcla y mismo cálculo que `evaluar` | |
| · `test_estadistica.py` (17) | Las tablas recalculadas desde los JSON de la v3 reproducen las 14 cifras publicadas (media y DE, a dos decimales); IC t; no se rellena el factorial; bootstrap pareado | |
| Controles de DeFM sin pesos | Pesos aleatorios: P3 y P5 varían; anulados como en D-05: P3 ~8e-10, P5 = 0; la extracción es la oficial bit a bit | `runs/defm/verificacion_defm_{aleatorio,degenerado}_sinteticos.{md,json}` |
| Huecos efectivos (sin datos; supone `depths` sin ceros) | Las aleatorias quitan exactamente la fracción nominal; la fraccional no quita ninguna celda; la binaria quita el 55 % en 1/8 con el 50 %; regiones 0.5 quita el 9.1 % de los píxeles (4.0-16.7) | `runs/degradaciones/huecos_efectivos.{md,json}` |
| Coste de extremo a extremo en CPU (fp32, lote 1, 480×640) | Parámetros y FLOPs (no dependen de los pesos ni de la máquina); latencia de CPU, solo indicativa | `runs/coste/coste_cpu_fp32.{md,json}` |
| Tablas desde los JSON de la v3 | `05_tablas_estadisticas.md` | `src/estadistica.py` |
| Humo de entrenamiento E1, E2c, E3c, E4c, E4cd y `diagnosticos_d06.py` | Corren de extremo a extremo con datos sintéticos | sesión (no se guardaron) |

## Pendiente (no se ejecutó: hace falta GPU, NYUv2 o los pesos oficiales)

| # | Qué | Comando | Necesita |
|---|---|---|---|
| P1 | Pruebas de GPU (2) | `pytest -q -m gpu` | CUDA |
| P2 | Verificación de DeFM con los pesos oficiales y NYUv2: normas, SHA-256, variabilidad, sensibilidad | `python tools/verificar_defm.py --nyuv2 --n 64` | pesos y NYUv2 |
| P3 | Fijar los SHA-256 de los dos checkpoints en `procedencia.SHA256_ESPERADO` | tras P2 | — |
| P4 | Reproducción de la v3 con el código nuevo (E1u, E2u y cualquier E4 con pesos) | `python tools/reproducir_historicos.py` | GPU, NYUv2 y pesos |
| P5 | Ceros en `depths` y huecos efectivos de «cruda» | `python tools/huecos_efectivos.py --nyuv2` | NYUv2 |
| P6 | Factorial E3c/E4c/E3cd/E4cd, E4cf/E4cfd, E2c y el control de E3c | `bash tools/cola_d06.sh 1` | GPU, unas 8-10 h |
| P7 | Coste en la GPU (bf16; latencia y VRAM) | `python src/coste.py` | GPU |
| P8 | Diagnósticos de MLCA en la validación (E4, E5 y la familia «c») | `python src/diagnosticos_d06.py ...` | pesos de esas corridas |
| P9 | Intervalos por remuestreo de imágenes | `python src/estadistica.py` (automático cuando existan `por_imagen.npz`) | las corridas de P6 |
| P10 | Revisar `requirements.txt` con numpy 2.4.4 (las pruebas corrieron con 2.1.3) | `pip install -r requirements.txt && pytest -q` | — |
| P11 | Comparar con el código real de la v2 las filas «por confirmar» de `03_comparacion_v2_v3.md` | lectura | el ZIP de la v2 |
| P12 | Predicciones del equipo para D-06, antes de abrir resultados | `DECISIONES.md` | el equipo |
