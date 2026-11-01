# Predicciones de la IA para D-07: resultado

Contraste de `predicciones/D07_claude.md` (cada sección escrita y registrada antes de lanzar su paso) con lo
medido el 2026-11-01. Test, media de 3 semillas; criterio con Holm de `tools/criterio_d06.py`. Las predicciones
originales no se tocan. ✔ dentro del intervalo; ✘ fuera, o con el sentido del efecto equivocado; ≈ fuera por
menos de 0.3 con el sentido correcto, o con una de dos partes fuera.

| # | Predicción | Medido | |
|---|---|---|---|
| Q1 | E11d completa 50.6 [49.8, 51.4]; − E11 = −0.45 [−1.2, +0.3] | 51.26; +0.20 | ✔ |
| Q2 | E11d sin 46.3 [45.0, 47.3] | 47.43 | ≈ (0.13 sobre el intervalo) |
| Q3 | E11d cruda 49.5 [48.3, 50.5] | 50.41 | ✔ |
| Q4 | E11d regiones 50.0 [49.2, 50.8] | 50.57 | ✔ |
| Q5 | E11d aleatoria_50 45 [41, 48] | 45.48 | ✔ |
| Q6 | E11d ruido_50cm 45 [42, 47.5] | 47.43 | ✔ |
| Q7 | Bajo el par E1u + E11: completa ≈ −1.6, cruda ≈ −2.3, sin ≈ −1.0 | −0.95, −1.36, +0.13 | ≈ (el signo en 2 de 3; menos distancia) |
| Q8 | lr 3e-3 (~55 %) | 1e-3 | ✘ |
| Q9 | E11d supera a E1u en todo salvo quizá «sin» (~70 %) | supera sin (+0.13); pierde con el 30 y el 50 % aleatorio (−0.25, −1.83) | ✘ («mejor modelo único» ✔) |
| R1 | E2cd completa 47.0 [46.2, 47.8]; − E2c = −0.5 | 48.05; +0.52 | ✘ (el dropout no costó: sumó) |
| R2 | E2cd cruda 46.4 [45.3, 47.3] | 47.66 | ✘ (0.36 por arriba) |
| R3 | E2cd regiones 46.6 [45.8, 47.3] | 47.26 | ✔ |
| R4 | E2cd aleatoria_50 43.5 [41.0, 45.5] | 44.58 | ✔ |
| R5 | E2cd sin 45.5 [44.5, 46.5] | 45.47 | ✔ |
| R6 | E2cd ruido_50cm 44.0 [42.0, 45.5] | 45.58 | ≈ (0.08 por arriba) |
| R7 | E2cd − E4cd: completa +1.8 [+0.8, +2.6], cruda +1.5 [+0.2, +2.5], sin −0.7 [−1.8, +0.4] | +2.83, +2.75, −0.69 | ≈ (el signo ✔ en las tres; sin ✔; las otras dos, por arriba) |
| S1 | E4e completa 49.7 [48.8, 50.6]; − E11 = −1.4 [−2.3, −0.5] | 49.39; −1.67 | ✔ |
| S2 | E3e − E4e = 0 exacto en completa y ruido | 0.00 | ✔ |
| S3 | E4e cruda 46.2 [44.5, 47.5]; E4e − E3e cruda −0.3 [−1.0, +0.4] | 47.78; +0.10 | ≈ (valor 0.28 por arriba; diferencia ✔) |
| S4 | E4e − E3e regiones +0.4 [0.0, +1.0] | +0.15 | ✔ |
| S5 | E4e sin 20 [8, 35], «colapsa como E11»; E3e sin 18 [6, 33] | 38.75; 29.36 | ✘ (E4e muy por arriba; E3e ✔) |
| S6 | E4ed completa 49.2 [48.2, 50.2]; − E11d = −1.4 [−2.4, −0.4] | 49.24; −2.02 | ✔ |
| S7 | E4ed sin 46.0 [44.8, 47.2]; − E11d = −0.3 [−1.3, +0.7] | 47.29; −0.14 | ≈ (valor 0.09 por arriba; diferencia ✔) |
| S8 | E4ed cruda 48.0 [46.8, 49.3] | 48.49 | ✔ |
| S9 | lr: E4e 3e-3 (~50 %); E4ed 1e-3 o 3e-3 | E4e 1e-3; E4ed 1e-3 | ≈ |
| S10 | La MLCA no supera a la suma en el codificador (E4e < E11, E4ed < E11d, más que el ruido) | −1.67 (las 3 semillas; sin bootstrap, E11 es histórico) y −2.02 («perjudica») | ✔ |

**Balance:** 14 ✔, 7 ≈, 5 ✘ (26 predicciones).

## Qué se sostuvo y qué no

- **El argumento de fondo se sostiene con profundidad completa, también en el codificador**: la MLCA da un
  resumen de toda la imagen y no la profundidad alineada posición a posición que aprovecha la suma; queda 1.7-2.0
  bajo ella (S1, S6, S10).
- **No vi la otra cara de ese mismo argumento.** Sin dropout, el resumen global también se degrada menos cuando la
  profundidad es mala: E4e − E11 = +3.2 y +6.0 con el 30 y el 50 % aleatorio, +4.2 con ruido de 50 cm y +23.0 sin
  profundidad. Casi todo viene de la atención (E3e, sin máscara, ya llega a 29.36 sin profundidad); la máscara
  suma +9.4 (S5). En la cabeza (D-06) no pasaba.
- **Volví a subestimar el dropout**: predije un costo pequeño con profundidad completa (Q1, R1) y no costó nada
  (+0.20 y +0.52). Corrijo la regla: con este protocolo, el dropout 0.3 es gratis.
- **lr**: dos de tres veces predije 3e-3 y salió 1e-3 (Q8, S9).

*Revisado el 2026-11-01 (segunda lectura): el balance contaba 27 predicciones en vez de 26; Q9 y S5 pasan de ≈ a
✘ con la regla de arriba, que antes no estaba escrita. La versión anterior queda en el commit 3f30edd.*
