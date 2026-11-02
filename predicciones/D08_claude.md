# Predicciones de la IA (Claude) para D-08

**Registradas:** 2026-11-02, antes de lanzar `tools/cola_d08.sh`, sin ningún resultado de E11r, E3er ni E4er.
**Quién:** Claude (Anthropic). La predicción del equipo está en `predicciones/D08_equipo.md`.

**Qué se sabía:** todos los resultados de D-06 y D-07, y cuánto falta en la profundidad cruda de entrenamiento: el
32.0 % de los píxeles del cuadro (13.9 % en el recorte estándar; casi todo el resto es el borde donde el Kinect no
cubre la imagen de color). La máscara binaria apaga el 32.6 % de las celdas de 1/16.

**D-08** = E11d, E3ed y E4ed (fusión también a la entrada de la última etapa del color, dropout 0.3) entrenados con
la profundidad cruda en lugar de la rellenada. Cambia solo la profundidad de entrenamiento: mismo orden de datos y
mismo dropout por semilla.

Cifras: mIoU en el test, media de 3 semillas; entre corchetes, intervalo de ~80 %. Referencias: E11d 51.26 / 50.41 /
45.48 / 47.43 (completa / cruda / 50 % aleatoria / sin); E4ed 49.24 / 48.49 / 44.10 / 47.29; E1u 47.31.

## Por qué

La brecha de la MLCA frente a la suma con profundidad completa (~2 puntos) viene del operador: un resumen de toda la
imagen no da la profundidad de cada posición (D-06, sección 6 del cuaderno 12). Entrenar con huecos cambia los datos,
no el operador, así que espero que la brecha siga. Lo que sí debería cambiar: con un tercio de los píxeles sin
medida al entrenar, las dos redes aprenden a convivir con huecos, y eso ayuda con la cruda y con huecos aleatorios.
Con profundidad completa hay un costo pequeño: los bordes rellenados nunca se vieron con valores al entrenar.

## Predicciones

| # | Qué | Predicción |
|---|---|---|
| H1 | E11r − E11d | completa −0.5 [−1.3, +0.3]; cruda +0.6 [−0.1, +1.3]; 50 % aleatoria +2.0 [+0.5, +3.5]; sin 0.0 [−0.5, +0.5] |
| H2 | E4er − E4ed | completa −0.2 [−1.0, +0.6]; cruda +0.8 [0.0, +1.6]; 50 % aleatoria +1.9 [+0.4, +3.4]; sin 0.0 [−0.5, +0.5] |
| H3 | E4er − E11r (la predicción del equipo: empate) | completa −1.8 [−2.6, −1.0]; cruda −1.7 [−2.6, −0.8]; 50 % aleatoria −1.5 [−3.0, 0.0]; sin −0.1 [−0.6, +0.4]. «Perjudica» en completa y cruda (~80 %); empate («no aporta») sin profundidad (~70 %). Probabilidad de empate en completa: ~15 % |
| H4 | E4er − E3er (la máscara, ahora activa al entrenar) | completa 0.0 [−0.7, +0.7]; cruda +0.6 [0.0, +1.2] («aporta» ~45 %); 50 % aleatoria +0.8 [−0.2, +1.8]; sin +0.8 [0.0, +1.6] |
| H5 | Frente a solo color (E1u 47.31) con 50 % aleatoria | E11r 47.5 [46.0, 49.0] y E4er 46.0 [44.5, 47.5]: E11r deja de quedar por debajo |
| H6 | lr | 1e-3 en las tres (~55 % cada una) |
| H7 | Conclusión | Entrenar con los huecos reales mejora con huecos y casi no cuesta con profundidad completa, pero la MLCA sigue bajo la suma cuando hay profundidad; empatan solo sin ella |

Me haría cambiar de idea: E4er − E11r > −0.5 con profundidad completa (los huecos al entrenar cierran la brecha del
operador), o E4er − E3er > +1 en la cruda (la máscara aprende algo útil cuando actúa al entrenar).
