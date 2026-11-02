# Predicción del equipo para D-08

**Registrada:** 2026-11-02, antes de lanzar `tools/cola_d08.sh` y sin ningún resultado de E11r, E3er ni E4er.
**Quién:** Joaquín Gonzalez Villarreal, por el equipo. Antes conocía los resultados de D-06 y D-07.

> «Creería que empata a la suma.»

Es decir: entrenada con los huecos reales del sensor, la MLCA con máscara (E4er) empata con la suma entrenada
igual (E11r).

**Cómo se contrasta** (criterio registrado de D-06, `tools/criterio_d06.py`): contraste E4er − E11r en completa,
cruda, 50 % aleatoria, regiones y sin profundidad. Se cuenta como empate «no aporta» (|dif.| < 0.3 y el IC 95 % por
imágenes incluye el 0). «Aporta», «perjudica» o «no concluyente» no son empate. El veredicto se da por condición.
