# Predicciones de la IA para D-06: resultado

Contraste de `predicciones/D06_claude.md` (commit b11ecf9, escrito antes de la cola) con lo medido el
2026-10-31/11-01. Test, media de 3 semillas; criterio con Holm de `tools/criterio_d06.py`. Las predicciones
originales no se tocan. ✔ dentro del intervalo; ✘ fuera; ≈ en el borde.

| # | Predicción | Medido | |
|---|---|---|---|
| P1 | E2c completa 47.9 [47.2, 48.8] | 47.53 | ✔ |
| P2 | E2c cruda 47.0 [46.0, 48.0]; cae 0.9 | 43.32; cae 4.21 | ✘ (el signo sí: cae más que E2) |
| P3 | E2c sin 40.5 [37.0, 43.5] | 20.33 | ✘ (el signo sí) |
| P4 | E4c − E3c = 0 exacto en completa y ruido | 0.00 | ✔ |
| P5 | E3c = E4c completa 46.0 [45.0, 47.2]; 1.9 [0.8, 3.0] bajo E2c | 44.99; 2.54 bajo E2c | ≈ (valor 0.01 bajo el intervalo; la brecha ✔) |
| P6 | Parte no constante de Δ en E4c < mitad de la de E2c en cada escala | 0.41 / 0.56 / 0.29 frente a 0.86 / 0.78 / 0.79 | ≈ (2 de 3 escalas) |
| P7 | E4c − E3c cruda +0.3 [0, 0.8]; «aporta» ~60 % | −0.63, «perjudica» | ✘ |
| P8 | E4c − E3c regiones +0.1 [−0.1, 0.4]; ~35 % | +0.63, «aporta» | ✘ (el signo sí) |
| P9 | E4c − E3c sin −16 [−38, +3]; E4c 22 [3, 42]; E3c 38 [30, 43] | +1.15 (no concluyente); E4c 14.86; E3c 13.71 | ✘ (E4c ✔, E3c ✘) |
| P10 | E4cd sin 45.0 [44.3, 45.8]; E3cd 44.3 [42.5, 45.5]; dif. +0.7 [−0.3, 2.0]; ~45 % | 46.16; 45.31; +0.85, «aporta» | ≈ (E4cd ✘ por arriba; E3cd y la diferencia ✔) |
| P11 | E4cd − E3cd cruda +0.3 [0, 0.8] | +0.01 | ≈ |
| P12 | Interacción máscara × dropout en «sin» fuertemente positiva | −0.30 ± 8.66 | ✘ |
| P13 | E4cd − E4c completa −0.2 [−0.8, 0.4] | +0.22 | ✔ |
| P14 | E4cf = E4c en completa, ruido y sin; resto \|dif.\| ≤ 0.3 (≤ 0.5 en aleatoria_50) | 0 exacto; −0.13, +0.02, −0.39 | ✔ |
| P15 | E3c derivado frente al entrenado (s0) \|dif.\| ≤ 0.3 | 0.71 | ✘ |
| P16 | E4c y E4cd eligen 3e-4; lr final ≤ 3e-4 | E4c 3e-3; E4cd 3e-4 (1e-4 probado, peor) | ≈ (la mitad) |

**Balance:** 4 ✔, 5 ≈, 7 ✘.

## Qué se sostuvo y qué no

- **Argumento de fondo (MLCA suma sobre todo un resumen global): se sostiene.** MLCA en la cabeza no gana nada
  con profundidad completa (44.99, lo mismo que E1) y queda 2.54 bajo la suma; su Δ es más constante que el de la
  suma (P6) y en 1/32 es 1.35 veces el rasgo de color.
- **Argumento 2 (quitar la máscara de golpe hunde a E4c más que a E3c): no se sostiene.** Sin dropout, los dos
  colapsan igual sin profundidad (14.86 y 13.71). El colapso no depende de la máscara sino de que la cabeza se
  apoya en ese desplazamiento casi constante de 1/32 (quitarlo: 43.18 → 13.76), que ninguna máscara repone.
- **Argumento 3 (la máscara ayuda con huecos parciales): a medias.** Ayuda con regiones grandes (+0.63; +1.61 y
  +2.28 con 0.75 y 1.0) y perjudica con los huecos reales del sensor (−0.63), finos y repartidos.
- **Subestimé al dropout.** Arregla también los huecos parciales (cruda +3.7/+4.4), no solo la ausencia total
  (P11 suponía lo contrario), y no cuesta nada con profundidad completa.
- **Subestimé la fragilidad de E2c.** Con los rasgos corregidos, la suma depende mucho más de la profundidad
  (quitar Δ: 46.38 → 37.44) sin ganar más que E2 (+0.22).
- **El no determinismo de la GPU es mayor de lo que supuse** (P15): 0.71 entre dos entrenamientos con la misma
  configuración y semilla. La derivación sigue siendo válida (la profundidad rellenada no tiene ceros, así que la
  máscara no actúa al entrenar), pero diferencias menores de ~0.7 entre modelos entrenados por separado, con
  una semilla, son ruido.
