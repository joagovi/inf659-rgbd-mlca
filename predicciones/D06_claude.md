# Predicciones de la IA (Claude) para D-06

**Registradas:** 2026-10-30, antes de lanzar `tools/cola_d06.sh 1`. El commit que añade este archivo es anterior
a cualquier corrida de la familia «c». **Quién:** Claude (Anthropic), a pedido de Joaquín. Son predicciones de la
IA, separadas de las del equipo, que van en `DECISIONES.md` (D-06) y se escriben sin ver resultados.

**Qué se sabía al escribirlas:** los resultados de la v3 y de D-05 (todos exploratorios), el paso 0 de D-06
(reproducción exacta de las cifras guardadas, DeFM verificado, huecos efectivos) y el código de `src/mlca.py`.
Ningún resultado de E2c, E3c, E4c, E3cd, E4cd, E4cf ni E4cfd.

Cifras: mIoU en el test, media de las 3 semillas; entre corchetes, un intervalo donde espero el valor con
~80 % de probabilidad. Referencias de la v3: E1 44.99; E2 47.31 (completa), 46.97 (cruda), 42.25 (sin);
E4 45.24 / 45.04 / 0.78; E5 45.48 / 45.45 / 45.74.

## El argumento de fondo

En `mlca.py` la salida para cada consulta es o_i = φ(q_i)ᵀS / φ(q_i)ᵀz: una media ponderada (pesos ≥ 0) de los
valores de profundidad de **toda** la imagen, sin codificación de posición. La consulta elige por contenido, no
por lugar, así que lo que MLCA suma a los rasgos de color es sobre todo un **resumen global de la profundidad de
la imagen**, poco alineado con cada píxel. La suma 1×1 de E2c, en cambio, pasa a cada posición la profundidad de
esa posición. De ahí salen casi todas las predicciones:

1. Con profundidad completa, la atención aprovecha menos la profundidad que la suma.
2. Como lo que suma MLCA es casi un desplazamiento común a toda la imagen, quitarlo de golpe (máscara sin
   ninguna celda válida) es un cambio grande para una cabeza que nunca lo vio: es el mismo mecanismo del
   colapso de E4, menos extremo porque ahora el desplazamiento cambia de imagen a imagen.
3. Con huecos parciales la máscara sí ayuda, porque la media se rehace solo con celdas válidas y sigue
   pareciéndose a lo que la cabeza vio al entrenar; sin máscara, las celdas de los huecos meten rasgos que la
   red nunca vio (se entrenó con la profundidad rellenada).

## Predicciones

| # | Qué | Predicción |
|---|---|---|
| P1 | E2c, completa | 47.9 [47.2, 48.8]; E2c − E2 = +0.6 [−0.1, +1.5] (E10 − E2u fue +0.65) |
| P2 | E2c, cruda | 47.0 [46.0, 48.0]; cae más que E2 (0.34): 0.9 [0.3, 1.8], porque C3 y C5 son más sensibles a los huecos que P4 |
| P3 | E2c, sin | 40.5 [37.0, 43.5]; cae más que E2, porque ahora cambian las tres escalas |
| P4 | E4c − E3c en completa y en los tres ruidos | **0.00 exacto** (por construcción: la máscara vale 1 en todas las celdas) |
| P5 | E3c = E4c, completa | 46.0 [45.0, 47.2]; por debajo de E2c en 1.9 [0.8, 3.0] (argumento 1) |
| P6 | Diagnóstico de D-06: parte que varía con la posición, ‖Δ − media(Δ)‖/‖Δ‖ | En E4c, menos de la mitad que en E2c en cada escala (argumento de fondo) |
| P7 | **E4c − E3c, cruda** (criterio) | +0.3 [0.0, +0.8]; cumple el criterio («aporta») con ~60 % de probabilidad |
| P8 | **E4c − E3c, regiones** (criterio) | +0.1 [−0.1, +0.4]; cumple el criterio con ~35 % |
| P9 | **E4c − E3c, sin** (criterio) | **negativa**: −16 [−38, +3]. E4c sin: 22 [3, 42]; E3c sin: 38 [30, 43] (argumento 2). Cumple el criterio con ~15 % |
| P10 | E4cd y E3cd, sin | E4cd 45.0 [44.3, 45.8] (≈ E1, como E5); E3cd 44.3 [42.5, 45.5]; E4cd − E3cd = +0.7 [−0.3, +2.0]; cumple el criterio con ~45 % |
| P11 | E4cd − E3cd, cruda | +0.3 [0.0, +0.8] (como P7: el dropout quita toda la profundidad, nunca una parte) |
| P12 | Interacción máscara × dropout en «sin» | **Fuertemente positiva**: el signo de la máscara pasa de negativo sin dropout (P9) a positivo con dropout (P10). La máscara solo sirve sin profundidad si la red aprendió a vivir sin ella |
| P13 | Costo del dropout con profundidad completa | E4cd − E4c = −0.2 [−0.8, +0.4] |
| P14 | E4cf frente a E4c (mismos pesos) | Idénticos en completa, ruido y sin (las dos máscaras coinciden). En las demás, \|diferencia\| ≤ 0.3, salvo aleatoria_50 (la binaria quita el 55 % de las celdas): ≤ 0.5 |
| P15 | E3c derivado frente al entrenado (semilla 0) | \|diferencia\| ≤ 0.3 en completa (solo el no determinismo de la GPU) |
| P16 | lr | E4c y E4cd eligen 3e-4, en el extremo inferior; la regla del extremo prueba 1e-4, y la lr final es ≤ 3e-4 (~70 %) |

## Lectura que haría con estas predicciones

- Si se cumplen P7-P9: la máscara binaria ayuda un poco con huecos parciales y perjudica sin profundidad cuando
  la red no se entrenó con fallos. El resumen del póster («con la máscara la salida vuelve a la de solo color»)
  sería cierto en los rasgos pero no en la precisión.
- Si se cumplen P10-P12: el resultado útil del factorial es la interacción, no el efecto de la máscara sola.
- Me haría cambiar de idea: E3c/E4c iguales o mejores que E2c con profundidad completa (contradice P5 y el
  argumento de fondo), o E4c sin profundidad cerca de E1 sin dropout (contradice P9 y el argumento 2).
