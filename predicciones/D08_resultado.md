# Predicciones para D-08: resultado

Contraste de `predicciones/D08_equipo.md` y `predicciones/D08_claude.md` (registradas el 2026-11-02, antes de lanzar
`tools/cola_d08.sh`) con lo medido ese mismo día. Test, media de 3 semillas; criterio con Holm de
`tools/criterio_d06.py` (`runs/rgbd/criterio_d08.txt`). Las predicciones originales no se tocan. ✔ dentro del
intervalo; ✘ fuera, o con el sentido del efecto equivocado; ≈ fuera por menos de 0.3 con el sentido correcto, o con
una parte fuera de dos o tres.

## La del equipo

> «Creería que empata a la suma.» (Joaquín, por el equipo)

Se contaba como empate «no aporta» en E4er − E11r: |dif.| < 0.3 y el IC por imágenes con el 0 dentro.

| Condición | E4er − E11r | Veredicto | |
|---|---|---|---|
| completa | −3.67 [−4.55, −2.74] | perjudica | ✘ |
| cruda | −3.12 [−3.93, −2.25] | perjudica | ✘ |
| 50 % aleatoria | −3.73 [−4.68, −2.74] | perjudica | ✘ |
| regiones | −3.67 [−4.40, −2.90] | perjudica | ✘ |
| sin | −2.04 [−2.64, −1.42] | perjudica | ✘ |

No se cumple en ninguna condición: entrenada con los huecos del sensor, la MLCA queda entre 2 y 3.7 puntos bajo la
suma, en las 3 semillas.

## La de la IA

| # | Predicción | Medido | |
|---|---|---|---|
| H1 | E11r − E11d completa −0.5 [−1.3, +0.3] | −0.63 | ✔ |
| H1 | E11r − E11d cruda +0.6 [−0.1, +1.3] | +0.98 | ✔ |
| H1 | E11r − E11d 50 % aleatoria +2.0 [+0.5, +3.5] | −0.03 | ✘ |
| H1 | E11r − E11d sin 0.0 [−0.5, +0.5] | −0.12 | ✔ |
| H2 | E4er − E4ed completa −0.2 [−1.0, +0.6] | −2.28 | ✘ |
| H2 | E4er − E4ed cruda +0.8 [0.0, +1.6] | −0.22 | ✘ |
| H2 | E4er − E4ed 50 % aleatoria +1.9 [+0.4, +3.4] | −2.37 | ✘ |
| H2 | E4er − E4ed sin 0.0 [−0.5, +0.5] | −2.02 | ✘ |
| H3 | E4er − E11r completa −1.8 [−2.6, −1.0] | −3.67 | ✘ (el sentido, bien) |
| H3 | E4er − E11r cruda −1.7 [−2.6, −0.8] | −3.12 | ✘ (el sentido, bien) |
| H3 | E4er − E11r 50 % aleatoria −1.5 [−3.0, 0.0] | −3.73 | ✘ (el sentido, bien) |
| H3 | E4er − E11r sin −0.1 [−0.6, +0.4] | −2.04 | ✘ |
| H3 | «Perjudica» en completa y cruda; empate sin profundidad | perjudica en las dos; sin profundidad, perjudica | ≈ |
| H4 | E4er − E3er completa 0.0 [−0.7, +0.7] | −2.05 | ✘ |
| H4 | E4er − E3er cruda +0.6 [0.0, +1.2] | −1.07 | ✘ |
| H4 | E4er − E3er 50 % aleatoria +0.8 [−0.2, +1.8] | −3.82 | ✘ |
| H4 | E4er − E3er sin +0.8 [0.0, +1.6] | −1.85 | ✘ |
| H5 | Con el 50 % aleatorio, E11r 47.5 [46.0, 49.0] y E4er 46.0 [44.5, 47.5]; E11r deja de quedar bajo E1u (47.31) | 45.46 y 41.73; E11r −1.85 bajo E1u | ✘ |
| H6 | lr 1e-3 en las tres | E11r 1e-3, E3er 1e-3, E4er 3e-3 | ≈ |
| H7 | Entrenar con los huecos mejora con huecos y casi no cuesta con profundidad completa; la MLCA sigue bajo la suma con profundidad y empatan sin ella | La suma mejora con los huecos del sensor (+0.98) y no con los aleatorios; cuesta −0.63 a la suma y −2.28 a la MLCA; la MLCA queda bajo la suma también sin profundidad | ✘ |

**Balance:** 3 ✔, 2 ≈, 15 ✘ (20 predicciones). Tampoco se dio ninguna de las dos señales que, según
`D08_claude.md`, harían cambiar de idea (E4er − E11r > −0.5 en completa; E4er − E3er > +1 en la cruda).

## Qué se sostuvo y qué no

- **Se sostuvo la dirección**: la brecha de la MLCA frente a la suma viene del operador, y entrenar con huecos no la
  cierra (H3, el sentido). La suma entrenada con la cruda gana en la cruda y pierde poco con profundidad completa
  (H1).
- **Falló el tamaño**: esperaba que la brecha siguiera en ~−1.8 y creció a −3.1 a −3.7 con profundidad (H3). Parte
  puede ser la lr: E4er quedó en 3e-3 (H6), su configuración más inestable entre semillas (DECISIONES, «Resultados
  de D-08»).
- **Falló que los huecos reales enseñaran a convivir con huecos aleatorios** (H1 y H5 con el 50 %): los del sensor
  tienen otro patrón (sobre todo el borde que el Kinect no cubre), y la red aprende ese.
- **Falló lo de la máscara**: esperaba que, activa al entrenar, ayudara algo en la cruda (H4) y que a la MLCA le
  sirviera entrenar con la cruda como a la suma (H2). Pasó lo contrario: E4er perdió ~2.3 parejo en todas las
  condiciones salvo la cruda, y sin máscara (E3er) le fue mejor en todas.
- **Lo que no vi venir**: entrenar con la cruda especializa la red en el sensor. E11r rinde más con la cruda que con
  la rellenada (51.39 frente a 50.63), y por eso es el mejor modelo único con el sensor real.
