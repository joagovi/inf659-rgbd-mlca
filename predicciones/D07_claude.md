# Predicciones de la IA (Claude) para D-07: E11d

**Registradas:** 2026-11-01, antes de lanzar `python src/entrenar_rgbd.py --completo E11d`. **Quién:** Claude
(Anthropic). Joaquín decidió el 2026-11-01 seguir sin las predicciones del equipo por ahora; estas hacen de
registro previo.

**Qué se sabía:** todos los resultados de D-06 (familia «c», diagnósticos y criterio) y de D-05, incluida la
prueba sin entrenar E1u+E11 con `relleno_interruptor`. Ningún resultado de E11d.

**E11d** = E11 (concatenación a la entrada de la última etapa del color, rasgos C3/P4/C5, `stage4` ajustada) +
modality dropout 0.3, con el generador propio de fallos (mismo orden de datos que E11 en cada semilla).

Cifras: mIoU en el test, media de 3 semillas; entre corchetes, intervalo de ~80 %. Referencias: E11 51.06
(completa), 47.71 (cruda), 49.19 (regiones), 33.39 (aleatoria_50), 15.75 (sin), 39.54 (ruido_50cm); E1u 47.30;
E1u+E11 con `relleno_interruptor` (dos modelos) 52.21 / 51.77 / 47.30 (completa / cruda / sin).

## Por qué

En la familia «c», el dropout no costó nada con profundidad completa (E4cd − E4c = +0.22, E3cd − E3c = +0.46) y
arregló casi del todo los huecos parciales (E4cd − E4c en cruda = +4.35), contra lo que predije en P11 de D-06.
En E11 la profundidad entra en el codificador, que tiene más capacidad que la cabeza para aprender a vivir sin
ella, pero comparte esa capacidad entre los dos regímenes: espero un costo pequeño con profundidad completa.

## Predicciones

| # | Qué | Predicción |
|---|---|---|
| Q1 | E11d, completa | 50.6 [49.8, 51.4]; E11d − E11 = −0.45 [−1.2, +0.3] |
| Q2 | E11d, sin | 46.3 [45.0, 47.3]: cerca de E1u, algo por debajo |
| Q3 | E11d, cruda | 49.5 [48.3, 50.5]; recupera la mayor parte de la caída de E11 (3.34) |
| Q4 | E11d, regiones | 50.0 [49.2, 50.8] |
| Q5 | E11d, aleatoria_50 | 45 [41, 48] |
| Q6 | E11d, ruido_50cm | 45 [42, 47.5] |
| Q7 | E11d frente a E1u+E11 con `relleno_interruptor` | Por debajo en completa (≈ −1.6), cruda (≈ −2.3) y sin (≈ −1.0). Probabilidad de que E11d iguale o supere al par en completa: ~5 % |
| Q8 | lr | 3e-3, como E11 (~55 %); si no, 1e-3 |
| Q9 | Lo mejor en un solo modelo | E11d es el mejor modelo único con robustez: supera a E1u en todas las condiciones salvo, quizá, «sin» (~70 %) |

Me haría cambiar de idea: E11d con profundidad completa más de 1.5 por debajo de E11 (el dropout sí cuesta en el
codificador), o E11d sin profundidad por debajo de 44 (no aprendió a vivir sin ella).

## E2cd (añadido el 2026-11-01, antes de lanzarlo y sin ningún resultado de E11d ni de E2cd)

**E2cd** = E2c + modality dropout 0.3 (generador propio: mismo orden de datos que E2c por semilla). Referencias:
E2c 47.53 / 43.32 / 44.79 / 30.38 / 20.33 / 39.26 (completa / cruda / regiones / aleatoria_50 / sin /
ruido_50cm); E4cd 45.21 / 44.91 / 44.93 / 42.23 / 46.16 / 43.80; E3cd 45.45 / 44.91 / 45.15 / 42.15 / 45.31 /
43.61. Sin máscara en la suma: sin profundidad, W·f_prof de una profundidad vacía es un desplazamiento fijo, que
el dropout enseña a la cabeza.

| # | Qué | Predicción |
|---|---|---|
| R1 | E2cd, completa | 47.0 [46.2, 47.8]; E2cd − E2c = −0.5 [−1.3, +0.3] |
| R2 | E2cd, cruda | 46.4 [45.3, 47.3] |
| R3 | E2cd, regiones | 46.6 [45.8, 47.3] |
| R4 | E2cd, aleatoria_50 | 43.5 [41.0, 45.5] |
| R5 | E2cd, sin | 45.5 [44.5, 46.5] |
| R6 | E2cd, ruido_50cm | 44.0 [42.0, 45.5] |
| R7 | E2cd − E4cd (suma frente a MLCA, los dos con dropout) | completa +1.8 [+0.8, +2.6]; cruda +1.5 [+0.2, +2.5]; sin −0.7 [−1.8, +0.4]. La suma gana con profundidad (~75 %); sin ella, empate o MLCA algo mejor |

## E4e, E3e y E4ed (añadido el 2026-11-01, antes de lanzarlos y sin ningún resultado de E11d, E2cd ni E4e)

**E4e** = E11 con MLCA (máscara binaria, umbral 0.5, en la rejilla de 1/16) en lugar de la suma W'·P4 a la
entrada de la última etapa del color; las sumas de la cabeza, como en E11. **E3e** = E4e sin máscara (derivado sin
entrenar: sin dropout, la máscara no actúa al entrenar). **E4ed** = E4e + dropout 0.3. Referencias: E10 (sin
nada en el codificador) 49.24; E11 51.06 / 47.71 / 49.19 / 15.75 (completa / cruda / regiones / sin).

**Por qué.** En E11 la profundidad del codificador es lo que más pesa (con la de otra imagen ahí, 51.06 → 31.09) y
es una profundidad alineada posición a posición. MLCA no la puede dar: cada salida es una media de los valores de
toda la imagen, elegida por contenido (D-06: en 1/16, solo el 56 % de su Δ varía con la posición, frente al 78 %
de la suma). Espero que E4e quede entre E10 y E11, más cerca de E10.

| # | Qué | Predicción |
|---|---|---|
| S1 | E4e, completa | 49.7 [48.8, 50.6]; E4e − E11 = −1.4 [−2.3, −0.5]; E4e ≥ E11 con ~10 % |
| S2 | E3e − E4e en completa y ruido | 0.00 exacto (derivado; la máscara vale 1) |
| S3 | E4e, cruda | 46.2 [44.5, 47.5]; E4e − E3e en cruda = −0.3 [−1.0, +0.4] (en D-06, −0.63) |
| S4 | E4e − E3e, regiones | +0.4 [0.0, +1.0] (en D-06, +0.63) |
| S5 | E4e, sin | 20 [8, 35]: colapsa como E11, sin dropout; E3e sin, 18 [6, 33] |
| S6 | E4ed, completa | 49.2 [48.2, 50.2]; E4ed − E11d = −1.4 [−2.4, −0.4] |
| S7 | E4ed, sin | 46.0 [44.8, 47.2]; E4ed − E11d en sin = −0.3 [−1.3, +0.7] |
| S8 | E4ed, cruda | 48.0 [46.8, 49.3] |
| S9 | lr | E4e, 3e-3 como E11 (~50 %); E4ed, 1e-3 o 3e-3 |
| S10 | Conclusión | La MLCA no supera a la suma tampoco en el codificador: E4e < E11 y E4ed < E11d por más que el ruido (~0.7) |

Me haría cambiar de idea: E4e ≥ E11 − 0.3 con profundidad completa (la atención sí aprovecha la profundidad donde
más rinde), o E4ed > E11d sin profundidad por más de 1 (la máscara sí da robustez en el codificador).
