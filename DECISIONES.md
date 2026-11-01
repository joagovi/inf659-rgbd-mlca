# Decisiones y predicciones

Registro de las decisiones del proyecto y de las predicciones escritas **antes** de cada experimento. Las
entradas no se borran: se revisan con una entrada nueva. Copia de lo que el equipo lleva en el repositorio
privado del curso (Deep Learning INF659, PUCP, 2026-2).

## D-01 · Comprobaciones previas, el canal min-max de DeFM y cómo se cuentan los huecos

**Fecha** 2026-10-21 · **Estado** ✅ cerrada · **Origen** el equipo, antes de entrenar E1 y E2.

- **Hechos.**
  - `notebooks/09_mlca_bloque`: sin profundidad, el bloque MLCA devuelve exactamente los rasgos de color; los
    huecos se excluyen de forma exacta; el coste crece con pendiente 1.14 frente a 2.00 de la softmax. DeFM
    ResNet-18 y EfficientViT-B1 dan rasgos de la misma resolución a 1/8, 1/16 y 1/32 de 480×640.
  - `notebooks/10_nyuv2_huecos`: en `rawDepths` falta el 30.7 % de la profundidad del test en la imagen
    completa y el 12.3 % dentro del recorte de Eigen. La diferencia es el marco donde el Kinect nunca mide,
    presente en todas las imágenes.
  - El tercer canal del preprocesado oficial de DeFM (`convert_to_3channel_metric`) normaliza cada imagen con
    su mínimo y su máximo, **contando los ceros de los huecos**. Con huecos, el mínimo pasa a 0 m y el canal de
    los píxeles válidos se desplaza una mediana de 0.21 (en NYUv2, `10_nyuv2_huecos` §4).
- **Decisión.**
  - El mínimo y el máximo del tercer canal se calculan **solo sobre los píxeles con medida**
    (`src/rgbd.py`, `entrada_defm`). Es una desviación declarada del código oficial de DeFM: sin ella, al pasar
    de profundidad completa a cruda cambiarían a la vez la información disponible y la codificación de lo
    medido.
  - La fracción de huecos se reporta **dentro del recorte de Eigen** (filas 45–470, columnas 41–600),
    indicándolo.
- **IA.** Escribió el módulo, los notebooks y el cargador, encontró la normalización con los huecos y propuso
  las dos decisiones; el equipo las aceptó.

## D-02 · Ruido gaussiano en la profundidad

**Fecha** 2026-10-25 · **Estado** 🟡 provisional · **Origen** el equipo: la propuesta incluye el ruido gaussiano
entre las condiciones de evaluación y faltaba definirlo.

- **Decisión.** Ruido gaussiano aditivo sobre la profundidad rellenada, en metros, con σ = 10, 20 y 50 cm
  (`ruido_10cm`, `ruido_20cm`, `ruido_50cm` en `src/rgbd.py`), recortado a un mínimo de 0.1 m para no crear
  huecos. Determinista por imagen. Los tres niveles siguen los de Liao et al. (σ = 0.1, 0.2 y 0.5), que allí
  son fracciones del rango normalizado [0, 1]; aquí se expresan en metros, así que no son la misma escala.
- **Cómo se mide.** Sobre los modelos finales ya entrenados, sin reentrenar
  (`python src/entrenar_rgbd.py --reevaluar`), que añade las condiciones nuevas a cada `resultado.json`.
  Recargar un modelo reproduce su resultado guardado (E2, semilla 0, completa: 47.54 en ambos casos).
- **Por qué importa.** El ruido conserva valores, así que la máscara de validez no lo detecta: MLCA no tiene
  ventaja de diseño frente a él (la «distinción técnica» de la propuesta).
- **IA.** Propuso la definición y escribió el código; los niveles quedan pendientes de confirmar por el equipo.

## D-03 · Profundidad como maestra (E7) y fusión con respaldo en la salida (E8)

**Fecha** 2026-10-28 · **Estado** 🟡 provisional · **Origen** el equipo, tras E4 y E5: la máscara sola no protege
a la red (E4 colapsa sin profundidad) y el *dropout* compra robustez con la ganancia (E5). Se registra antes de
entrenar.

- **Pruebas sin entrenar** (`src/combinar_e1_e2.py`), con los modelos ya entrenados de E1 y E2: interruptor por
  imagen según la fracción de profundidad válida; mezcla por píxel p = (1 − α)·p_E1 + α·p_E2, con α = fracción
  válida en una ventana de k×k celdas de 1/8; y relleno de los huecos con el píxel válido más cercano antes de
  E2. El umbral (0.3, 0.5, 0.7, 0.9) y k (1, 3, 7, 15, 31) se eligen en la validación por escenas con la media
  de todas las condiciones; el test se evalúa una vez.
- **E1u**: E1 con la última etapa de EfficientViT («stage4») entrenable, con una lr 10 veces menor que la de la
  cabeza y BN con sus estadísticas de ADE20K. Es la referencia justa para E7.
- **E7** (profundidad como maestra; opción (c) de la tesis): E1u más una cabeza auxiliar que predice el log de la
  profundidad, copia de la cabeza de ADE20K con salida 1 iniciada en cero. Pérdida invariante a la escala de cada
  imagen (Eigen et al., 2014), porque el aumento amplía la imagen sin cambiar la profundidad. Se descarta al
  inferir: E7 no usa profundidad en el test. lr = la elegida para E1u; el peso de la pérdida auxiliar (1, 3, 10)
  se elige en la validación.
- **E8** (fusión con respaldo en la salida; opción (b′) de la tesis): dos expertos sobre los mismos codificadores
  congelados, la cabeza de color (como E1) y la de fusión (como E2), mezclados por píxel con
  α = σ(w·[m₁, m₅, m₁₅] + b)·m₁₅, donde m_k es la fracción válida en una ventana de k celdas (`rgbd.Compuerta`).
  Sin ninguna medida en la ventana grande, α = 0 y la salida es exactamente la del experto de color (comprobado:
  diferencia máxima 1e-6 en el log de la probabilidad). Pérdida: mezcla + experto de color + experto de fusión,
  este último solo en los píxeles con medida, para que no aprenda a prescindir de la profundidad (lo que le pasó
  a E5). Al entrenar, por muestra: sin profundidad con prob. 0.15, rectángulos de severidad U(0, 1) con prob.
  0.35 y completa en el resto.
- **E2h** (control de E8): E2 con la misma corrupción al entrenar, para separar el efecto de la compuerta del de
  ver huecos al entrenar.
- **Limitación declarada:** E8 y E2h ven rectángulos al entrenar, así que la condición «regiones» no es nueva
  para ellos; «cruda», «aleatoria» y «ruido» sí lo son.
- **Comprobado antes de lanzar:** con el código nuevo, E1 y E2 recargados reproducen sus resultados guardados
  (45.1702, 47.5440 y 42.3615); en E1u solo la última etapa recibe gradiente.
- **Predicciones:** E7 y E8 se lanzan sin predicción registrada, a pedido del equipo; se puede registrar antes de
  abrir los resultados.
- **IA.** Propuso los diseños y los valores fijados de antemano y escribió el código.

## D-04 · Controles de ajuste y de ensamble; E2u y E9

**Fecha** 2026-10-29 · **Estado** 🟡 provisional · **Origen** los resultados de D-03 (al final de este archivo): E1u
iguala a E2 sin usar profundidad, y la ventaja de E8 con profundidad completa se explica por promediar dos
cabezas. Se registra antes de entrenar.

- **Línea base de color: E1u**, no E1. Para un modelo con dos cabezas, o para dos modelos combinados, la
  comparación justa incluye el promedio de dos modelos de color de semillas distintas (control del ensamble).
- **E2u**: E2 con la última etapa del codificador de color ajustada como en E1u (lr 10 veces menor). Mide si la
  profundidad suma sobre el ajuste en un solo modelo.
- **Pruebas sin entrenar con E1u + E2u** (`src/combinar.py`; `combinar_e1_e2.py` se conserva): las de D-03 más
  un tope a ∈ {0.5, 0.6, 0.7, 0.8, 1} para α, α = a · m_k. Con a < 1 los dos modelos también se promedian con
  profundidad completa, como hace la compuerta de E8. El umbral del interruptor, (k, a) de la mezcla y a del
  relleno se eligen en la validación por escenas con la media de todas las condiciones. Incluye el control: el
  promedio de dos E1u de semillas distintas.
- **E9**: E8 con el experto de color **congelado e igual al E1u ya entrenado** de la misma semilla (en la
  validación, el E1u de la validación, que no vio esas escenas). La rama de fusión tiene su propia copia de la
  última etapa, ajustable (lr 10 veces menor), y su propia cabeza; compuerta y corrupción al entrenar como en E8.
  Pérdida: mezcla + experto de fusión en los píxeles con medida. Sin profundidad, E9 es exactamente E1u: la
  garantía queda a nivel de red, que es lo que el bloque de E4 no daba.
- **Comprobado antes de lanzar:** E8 recargado con el código nuevo reproduce su resultado (semilla 0: 48.5345
  con profundidad completa y 44.2995 sin ella); E9 recién construido da sin profundidad 47.2617, frente a 47.2616
  de E1u (diferencia máxima 1.9e-6 en el log de la probabilidad); en E9 solo reciben gradiente la última etapa de
  la rama de fusión, la fusión, su cabeza y la compuerta, y el experto queda en modo evaluación.
- **Orden:** diagnósticos de D-03 (`src/diagnosticos_d03.py`) → E2u → E1u + E2u → E9 → E1 + E2 con el tope, para
  el registro.
- **Para después:** módulos de fusión del estado del arte (CMX, ESANet), con los mismos controles de ajuste y de
  ensamble, para no confundir capacidad o promedio con mejor fusión.
- **Predicciones:** se pueden registrar antes de abrir los resultados.
- **IA.** Encontró las dos confusiones (ajuste y ensamble), propuso los controles y los diseños, y escribió el
  código.

## D-05 · P3 y P5 de DeFM son constantes; E10 y E11

**Fecha** 2026-10-29 · **Estado** 🟡 provisional · **Origen** los diagnósticos de D-04 (al final de este archivo).
Se registra antes de entrenar.

- **Hallazgo.** En los pesos publicados de DeFM ResNet-18 (`leggedrobotics/defm`, `defm_resnet18`), las salidas
  P3 y P5 de la BiFPN (`dense_bifpn`) no dependen de la entrada: las convoluciones de salida de esos niveles
  (`fpn.blocks.0.convs.0` y `.2`) tienen pesos de norma 1.3e-9, y en P5 también los pesos de fusión
  (`fuse_bottomup.1`) son 0. Entre dos imágenes distintas, P3 cambia 4e-10 (relativo), P5 nada y P4 0.98. **Hasta
  E9, la profundidad solo entró en 1/16;** en 1/8 y 1/32 la fusión sumó una constante aprendida. En MLCA (E4, E5)
  también: la atención sobre claves y valores constantes da una constante. Afecta a todo uso de `dense_bifpn` de
  DeFM ResNet-18. El repositorio de DeFM todavía no publica cómo hace la segmentación (`segmentation/` figura como
  TODO en su README).
- **E10**: E2u con rasgos que dependen de la entrada en las tres escalas. En 1/8 y 1/32, las etapas del ResNet de
  DeFM (C3, 128 canales; C5, 512) con LayerNorm por posición sin parámetros (DeFM normaliza así sus salidas
  densas, con parámetros); en 1/16, P4, como hasta ahora (`rgbd.rasgos_defm`). Lo demás, como E2u; +98 k
  parámetros entrenables, porque la W de 1/32 pasa de 128 a 512 entradas.
- **E11** (dónde entra la profundidad): E10 con la profundidad también dentro del codificador de color. La última
  etapa lee f_1/16 + W'·P4, con W' propia (1×1, iniciada en cero), como la rectificación de CMX antes de la etapa
  siguiente. Las sumas que lee la cabeza no cambian; con W' = 0 es E10; +16 k parámetros.
- **Cómo se evalúan:** solos, en todas las condiciones, y con E1u de respaldo (`combinar.py`, la misma selección
  en la validación), frente a E1u + E2u (49.70) y al control E1u + E1u (48.50).
- **Barrido de lr:** E2u eligió 3e-3 con la validación todavía subiendo (45.34 / 46.93 / 47.76). En E2u, E10 y E11,
  si la mejor lr queda en un extremo, se prueba la siguiente de (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2) hacia ese
  lado, hasta que quede dentro. Si E2u cambia de lr, se repiten sus 3 semillas y E1u + E2u; el resultado anterior
  se conserva (`combinar.py` lo guarda aparte).
- **Comprobado antes de lanzar:** con el código nuevo, E1u, E2, E2u, E4 y E9 reproducen sus resultados guardados
  (semilla 0; por ejemplo, E2u 48.6364 con profundidad completa y 36.6998 sin ella, E4 45.5647 y 45.1796 con la
  cruda); el P4 de `rasgos_defm` es idéntico al de `dense_bifpn`; C3 y C5 cambian con la imagen (1.18 y 0.93,
  relativo); E10 y E11 recién construidos dan la misma salida que E2u, y E11 con W' = 0 la misma que E10; solo
  reciben gradiente la última etapa del color, la fusión (y W' en E11) y la cabeza.
- **Orden:** E2u (extremo del barrido) → E10 → E1u + E10 → E11 → E1u + E11 → E1u + E2u, si E2u cambió de lr.
- **Para después:** E4 y E5 (MLCA) con los rasgos de E10, si el equipo mantiene MLCA en el informe; módulos del
  estado del arte en la ubicación que gane; una compuerta aprendida, solo con datos que los expertos no hayan visto
  (resultados de D-04).
- **Predicciones:** se pueden registrar antes de abrir los resultados.
- **IA.** Encontró las salidas constantes de DeFM con la prueba de la profundidad de otra imagen, propuso E10 y
  E11 y escribió el código.

## Predicciones de E1 y E2 (NYUv2, 40 clases)

Registradas el 2026-10-21 en el repositorio del curso antes de entrenar. Se habían visto `09_mlca_bloque` y
`10_nyuv2_huecos`, ningún entrenamiento. Texto tal como se dijo.

| Experimento | Qué esperábamos | Por qué | Resultado (test, 3 semillas; `11_e1_e2_nyuv2` §1) |
|---|---|---|---|
| E1: EfficientViT-B1 solo color, mIoU | «será alrededor de 50 quizá 40 pero difícil más de 50» | «dado que son 40 clases y es one shot» | 44.99 ± 0.30 (lr 1e-3, elegida en la validación por escenas: 43.43) |
| E2: EfficientViT + DeFM por concatenación, profundidad completa, frente a E1 | «con la profundidad creo que va subir unos 5 puntos» | no se escribió | 47.31 ± 0.26, +2.32 sobre E1 (+2.21 a +2.39 por semilla); con huecos reales 46.97 ± 0.52, +1.98 |
| E2 con 50 % de huecos | «2 a 3 puntos debe caer con 50 por ciento de huecos» (se entiende frente a E2 con profundidad completa) | no se escribió | 43.90 ± 0.12: −3.42 frente a E2 completa y −1.09 frente a E1. Con 30 %: −2.50 y −0.17. Sin profundidad: 42.25 ± 0.45, −2.74 frente a E1 |

E2 «sin profundidad» recibe un mapa de profundidad vacío; entrenada siempre con profundidad y sin máscara, no
puede distinguir la ausencia de un valor medido, por lo que no equivale a E1, que no tiene rama de profundidad.

## Predicciones de E4 (MLCA, NYUv2, 40 clases)

Registradas el 2026-10-27 antes de entrenar E4. Se conocían los resultados de E1 y E2 de este repositorio; no se
habían visto los resultados de E4 de ninguna otra implementación. Texto tal como se dijo.

| Condición | Qué esperamos | Por qué | Resultado |
|---|---|---|---|
| Profundidad completa | «subirá entre 3 a 5 puntos con profundidad completa» (aclarado: sobre E2, 47.31 → 50.3 a 52.3) | no se escribió | 45.24 ± 0.29: −2.07 frente a E2 y +0.25 frente a E1 |
| 50 % de huecos | «en E2 creo que va subir levemente 1 punto y respecto a E1 si va caer un poco porque 50% es bastante hueco» (E2: 43.90; E1: 44.99) | «50% es bastante hueco» | 41.89 ± 1.40: −2.01 frente a E2 y −3.10 frente a E1 (con 30 %: 42.04 ± 0.41) |
| Sin profundidad | «en realidad va a salir mejor que E1 [...] también entonces debería salir mejor a E2» (E1: 44.99; E2: 42.25) | «el color llegará intacto, porque la máscara saca esos rasgos» | 0.78 ± 1.22 (0.03, 0.12 y 2.19 por semilla): la red colapsa |

Una primera versión de la predicción sin profundidad («peor que E1, parecido a E2») se cambió antes de entrenar,
después de que la IA explicara que el bloque MLCA suma exactamente 0 cuando no hay profundidad válida, mientras
que en E2 los rasgos de DeFM de una imagen vacía no son cero.

Medido el 2026-10-28 (test, 3 semillas; lr 3e-4, elegida en la validación por escenas: 44.15, frente a 42.81 con
1e-3 y 42.31 con 3e-3; 3e-4 es el extremo inferior del barrido). Resto de condiciones de E4: cruda 45.04 ± 0.19,
regiones 44.87 ± 0.39, ruido de 10/20/50 cm 45.05 / 44.90 / 44.18. **Diagnóstico del colapso sin profundidad**
(semilla 0, 41 imágenes del test): el bloque funciona como se diseñó (sin profundidad suma exactamente 0), pero lo
que suma con profundidad es grande frente a los rasgos de color: |Δ|/|f_color| = 0.74, 0.89 y **9.43** a 1/8, 1/16
y 1/32 (en E2: 0.82, 0.67 y 1.36). La cabeza se entrenó siempre con ese término presente; al quitarlo recibe rasgos
que nunca vio y predice casi una sola clase (97 % de los píxeles en 3 clases). Que el bloque aporte cero no hace
que la red se comporte como E1: falta que la cabeza vea, al entrenar, casos sin profundidad (E5).
## Predicción de E5 (E4 + modality dropout de 0.3)

Registrada el 2026-10-28, con E5 ya en entrenamiento y antes de ver ningún resultado suyo (tampoco el E5 de otra
implementación). Texto tal como se dijo.

| Experimento | Qué esperamos | Por qué | Resultado |
|---|---|---|---|
| E5 frente a E4 y a E1, en todas las condiciones | «el dropout hace que todo se mantenga igual o caída y según E1 creo que caerá» (se entiende: E5 igual o por debajo de E4, y por debajo de E1) | «dado los resultados anteriores» | completa 45.48 ± 0.21, cruda 45.45 ± 0.20, aleatoria 10/30/50 % 45.32 / 44.46 / 44.25, regiones 45.39 ± 0.07, sin 45.74 ± 0.10, ruido 10/20/50 cm 45.40 / 45.29 / 44.79 |

Aclaración del equipo, el mismo día y antes de ver resultados: «lo que decía era E5 por encima ligeramente o igual
que E4, pero debajo de E1». Sustituye a la interpretación anterior («igual o por debajo de E4»). Con profundidad
completa, E4 (45.24) ya está por encima de E1 (44.99): en esa condición las dos partes no pueden cumplirse a la vez.

Segunda aclaración, el mismo día y antes de ver resultados, para completa y cruda (donde E4 ya supera a E1):
«parece que con profundidad mejora ligeramente, entonces en ese caso que falta pienso que superará a ambos»
(E5 por encima de E4 y de E1). En las demás condiciones sigue la anterior: igual o ligeramente por encima de E4 y
por debajo de E1.

Medido el 2026-10-28 (test, 3 semillas; lr 3e-4, elegida en la validación por escenas: 43.78, frente a 43.34 con
1e-3 y 42.25 con 3e-3; de nuevo el extremo inferior del barrido). E5 no colapsa: entre todas las condiciones va de
44.25 a 45.74, y sin profundidad queda 0.75 por encima de E1. Pero aprovecha poco la profundidad: con profundidad
completa gana 0.49 a E1 (E2: 2.32) y queda por debajo de su propio resultado sin profundidad. **Diagnóstico**
(semilla 0, 41 imágenes): lo que suma MLCA frente al color es |Δ|/|f| = 0.01, 0.46 y 0.02 a 1/8, 1/16 y 1/32 (E4:
0.74, 0.89 y 9.43). Con el dropout, la red apagó casi del todo la fusión a 1/8 y 1/32 y solo la mantiene a 1/16.
Ruido en E1 y E2, ya evaluado: E1 44.99 en los tres niveles; E2 46.66 / 46.25 / 45.25 con 10 / 20 / 50 cm.

## Resultados de D-03 (E1u, E7, E8, E2h y pruebas sin entrenar)

Medido el 2026-10-29 (test, 3 semillas). No se registró predicción: el equipo pidió ver los resultados directamente.
Elegido en la validación por escenas: E1u lr 1e-3 (43.87 / 45.47 / 43.14 con 3e-4 / 1e-3 / 3e-3); E7 peso
auxiliar 1 (45.61, frente a 45.38 con 3 y 45.41 con 10); E8 lr 3e-3 (45.31 / 45.97 / 46.58: extremo superior del
barrido, todavía subiendo); E2h lr 1e-3 (44.94 / 46.04 / 44.95); pruebas sin entrenar, umbral 0.7 y ventana 15.

| Modelo | completa | cruda | aleatoria 10 / 30 / 50 % | regiones | sin | ruido 10 / 20 / 50 cm |
|---|---|---|---|---|---|---|
| E1 (referencia) | 44.99 ± 0.30 | 44.99 | 44.99 / 44.99 / 44.99 | 44.99 | 44.99 ± 0.30 | 44.99 / 44.99 / 44.99 |
| E2 (referencia) | 47.31 ± 0.26 | 46.97 | 46.66 / 44.82 / 43.90 | 46.58 | 42.25 ± 0.45 | 46.66 / 46.25 / 45.25 |
| E1u | 47.31 ± 0.10 | igual | igual | igual | igual | igual |
| E7 | 47.28 ± 0.54 | igual | igual | igual | igual | igual |
| E8 | 48.17 ± 0.33 | 47.08 | 47.62 / 45.81 / 45.11 | 47.70 | 44.00 ± 0.29 | 47.74 / 47.21 / 46.21 |
| E2h | 46.98 ± 0.22 | 46.63 | 46.55 / 44.60 / 43.95 | 46.75 | 44.59 ± 0.20 | 46.48 / 46.04 / 45.08 |
| Interruptor E1/E2 | 47.31 ± 0.26 | 45.95 | 46.67 / 44.98 / 44.97 | 46.58 | 44.97 ± 0.30 | 46.66 / 46.25 / 45.25 |
| Mezcla E1/E2 | 47.31 ± 0.26 | 47.35 | 46.97 / 46.13 / 46.07 | 46.99 | 44.97 ± 0.30 | 46.66 / 46.25 / 45.25 |
| Relleno + E2 | 47.31 ± 0.26 | 47.13 | 47.31 / 47.31 / 47.32 | 47.08 | 42.28 ± 0.46 | 46.66 / 46.25 / 45.25 |

La desviación entre semillas no pasa de 0.54 en ninguna celda. E1u y E7 no usan profundidad: dan lo mismo en
todas las condiciones.

**Diagnósticos** (exploratorios, sobre el test; `src/diagnosticos_d03.py`; no sirven para elegir modelos):

- **E1u frente a E2.** Ajustar la última etapa del color da lo mismo que añadir profundidad a un color congelado
  (47.31 ambos). Por clase, las dos ganancias sobre E1 se parecen solo en parte (correlación 0.53): la profundidad
  ayuda donde el ajuste no (silla +4.9 frente a +0.8, cojín +5.5 / +0.7, bañera +4.5 / −0.7, lámpara +3.9 / −0.3,
  piso +3.3 / +0.3, techo +2.2 / −1.5) y el ajuste donde la profundidad menos (cómoda +9.0 / +4.0, lavabo
  +6.8 / +2.5, puerta +6.2 / +2.2).
- **E8 con profundidad completa es un promedio de dos cabezas.** La compuerta vale en media 0.59 (no 1): mezcla
  48.17, experto de fusión solo 47.08, experto de color solo 44.00. Promediar dos modelos de color de semillas
  distintas, sin profundidad, da 46.60 con E1 (+1.6 sobre uno solo) y 48.50 con E1u. Color más fusión: E1u + E2
  da 49.11 / 49.26 / 49.07 con α = 0.5 / 0.6 / 0.7, frente a 48.50 de E1u + E1u y 48.27 de E1u + E1 (α = 0.5). Con
  el promedio controlado, la profundidad aporta unos 0.8 (cifra exploratoria: la de la propuesta saldrá de
  `src/combinar.py`, con α elegido en la validación).
- **El experto de color de E8 (44.00) queda por debajo de E1 (44.99)** aunque la garantía de la mezcla es exacta:
  E8 eligió lr 3e-3, que favorece a la fusión, y con esa lr una cabeza de color pierde 1.45 en la validación
  (E1: 41.98 con 3e-3 frente a 43.43 con 1e-3).
- **E7 aprendió profundidad, pero no le sirvió.** Su cabeza auxiliar, con la escala alineada por imagen: SI-RMSE
  del log 0.169, AbsRel 0.135 y δ < 1.25 del 82.7 % (una constante: 0.357, 0.321 y 46.0 %). La segmentación queda
  igual que E1u, también en las clases donde la profundidad de entrada ayuda.
- **Huecos al entrenar (E2h frente a E2):** +2.3 sin profundidad, −0.3 con profundidad completa y nada con huecos
  aleatorios (los rectángulos de entrenamiento no se transfieren a píxeles sueltos). La ventaja de E8 con huecos
  aleatorios viene de la compuerta.
- **«Aleatoria» es una condición fácil:** el relleno con el vecino válido devuelve el resultado con profundidad
  completa (47.31 con el 10, el 30 y el 50 %). Las que separan los métodos son «regiones», «cruda» y «sin».

Siguiente paso: D-04.

## Resultados de D-04 (E2u, E9 y pruebas sin entrenar con E1u + E2u)

Medido el 2026-10-29 (test, 3 semillas). No se registró predicción. Elegido en la validación por escenas: E2u lr
3e-3 (45.34 / 46.93 / 47.76 con 3e-4 / 1e-3 / 3e-3: extremo superior del barrido, todavía subiendo); E9 lr 3e-4
(45.96 / 45.85 / 45.55: extremo inferior, casi plano); pruebas sin entrenar, con E1u + E2u y con E1 + E2: umbral
0.7, mezcla con ventana 15 y tope 0.6 (la ventana casi no importa: 46.93 en la validación con 1, 3, 7, 15 o 31) y
relleno con a = 0.5.

| Modelo | completa | cruda | aleatoria 10 / 30 / 50 % | regiones | sin | ruido 10 / 20 / 50 cm |
|---|---|---|---|---|---|---|
| E1u (referencia) | 47.31 ± 0.10 | igual | igual | igual | igual | igual |
| E2u | 48.59 ± 0.15 | 47.71 | 47.89 / 44.94 / 43.90 | 47.72 | 36.84 ± 0.70 | 47.88 / 47.12 / 45.47 |
| E9 | 47.76 ± 0.09 | 47.72 | 47.75 / 47.64 / 47.57 | 47.73 | 47.31 ± 0.10 | 47.74 / 47.71 / 47.65 |
| E1u + E1u (control) | 48.50 ± 0.08 | igual | igual | igual | igual | igual |
| Interruptor E1u/E2u | 48.52 ± 0.14 | 47.58 | 47.84 / 46.16 / 47.27 | 47.67 | 47.27 ± 0.10 | 47.83 / 47.07 / 45.44 |
| Mezcla E1u/E2u | 49.70 ± 0.07 | 49.24 | 49.31 / 48.08 / 47.82 | 49.41 | 47.27 ± 0.10 | 49.22 / 48.71 / 47.61 |
| Relleno + E1u/E2u | 49.71 ± 0.01 | 49.64 | 49.71 / 49.72 / 49.72 | 49.55 | 45.55 ± 0.11 | 49.37 / 48.99 / 48.21 |
| E1 + E1 (control) | 46.60 ± 0.12 | igual | igual | igual | igual | igual |
| Mezcla E1/E2 | 47.63 ± 0.32 | 47.01 | 47.24 / 46.25 / 45.76 | 47.20 | 44.97 ± 0.30 | 47.33 / 47.09 / 46.53 |
| Relleno + E1/E2 | 47.31 ± 0.33 | 47.28 | 47.31 / 47.31 / 47.31 | 47.22 | 45.54 ± 0.36 | 47.10 / 46.92 / 46.48 |

Las pruebas sin entrenar promedian probabilidades: ahí E1u da 47.27 y E1 44.97 (con logits, 47.31 y 44.99). Los
controles no usan profundidad. La desviación entre semillas no pasa de 0.70 (E2u sin profundidad).

**Diagnósticos** (exploratorios, sobre el test salvo donde se dice; `src/diagnosticos_d04.py`; no sirven para
elegir modelos):

- **La profundidad sobre el ajuste, en un solo modelo.** E2u − E1u = +1.28 con profundidad completa (+1.37, +1.00
  y +1.46 por semilla), frente a +2.32 de E2 sobre E1 con el color congelado. Sin profundidad, E2u cae a 36.84
  (E2: 42.25): ajustar el color junto con la profundidad lo hace más dependiente de ella.
- **Por clase.** E2u supera a E1u en 30 de 40 clases, sobre todo en clases de forma (cojín +4.7, techo +3.6,
  inodoro +3.3, piso +3.1, silla +2.6, cama +2.3), y pierde donde el ajuste del color había ganado más (cómoda
  −4.9, refrigerador −4.2, puerta −1.9): en cómoda, E2u (50.8) queda como E2 (50.7) y no como E1u (55.6).
  Correlación entre E2u − E1u y E1u − E1: −0.60. La pizarra (+8.8) es inestable: E1u y E2 pierden 7.0 y 11.2
  sobre E1.
- **La cifra para la propuesta es la mezcla E1u + E2u:** 49.70 ± 0.07 con profundidad completa, +1.20 sobre el
  control E1u + E1u y +2.43 sobre E1u, con tope y ventana elegidos en la validación (la cifra de 0.8 de D-03 se
  eligió en el test). Sin profundidad es E1u. Supera al control con profundidad completa, cruda, por regiones,
  con el 10 % de huecos aleatorios y con ruido de 10 y 20 cm; queda por debajo con el 30 y el 50 % de huecos
  aleatorios, sin profundidad y con ruido de 50 cm, sin bajar nunca de E1u. El relleno supera al control con
  todos los huecos (49.55 a 49.72), pero no sin profundidad (45.55: no hay de dónde rellenar) ni con ruido de
  50 cm. Coste: E1u y E2u comparten el tronco congelado (entrada y etapas 1 a 3), así que la mezcla es una pasada
  del tronco, dos últimas etapas, dos cabezas y DeFM: la misma red que E9. El control es lo mismo sin DeFM ni
  fusión.
- **E9 deja sin usar la profundidad que tiene.** Su compuerta vale en media 0.14 con profundidad completa (E8:
  0.59). Su experto de fusión solo da 47.76, y con los mismos dos expertos y α fijo = 0.6 · m_15 la mezcla daría
  48.84 (+1.1 sobre E9). En la validación pasa lo mismo, y peor cuanto mayor es la lr: α medio 0.18 / 0.09 /
  0.015 con 3e-4 / 1e-3 / 3e-3, aunque el experto de fusión supera al de color con 1e-3 y 3e-3 (46.79 y 45.96
  frente a 45.39) y el α fijo daría 46.33 / 47.32 / 48.13. La causa está en las imágenes de entrenamiento, que
  es donde se ajusta la compuerta: ahí el experto de color (E1u, entrenado 60 épocas con ellas) da 86.25 y el de
  fusión 75.69, y lo correcto es confiar en el color. Una compuerta aprendida necesita datos que los expertos no
  hayan visto, como el apilamiento (Wolpert, 1992); la mezcla sin entrenar elige su tope en la validación por
  escenas.
- **Escalas de la fusión.** Las sumas W·f_prof no son correcciones pequeñas: su norma, relativa a la del rasgo
  de color, es 0.85 / 0.66 / 1.40 en E2 y 9.3 / 1.53 / 2.79 en E2u (1/8, 1/16, 1/32). Quitar la de 1/8 o la de
  1/32 hunde a E2u (5.3 y 1.0), y quitar las tres deja a E2 en 9.3 y a E2u en 0.9: la cabeza se ajustó a
  recibirlas. Pero con la profundidad de otra imagen del test en 1/8 o en 1/32 el mIoU no cambia ni en el tercer
  decimal; en 1/16 cae de 47.31 a 42.49 (E2) y de 48.59 a 40.47 (E2u). La causa, en D-05: en los pesos publicados
  de DeFM ResNet-18, P3 y P5 son constantes.

Siguiente paso: D-05.

## Resultados de D-05 (E10, E11 y pruebas sin entrenar con E1u)

Medido el 2026-10-29 (test, 3 semillas). No se registró predicción. Elegido en la validación por escenas: E2u
sigue en 3e-3 (con 1e-2, 36.36), así que su resultado y E1u + E2u no cambian; E10 lr 1e-3 (45.45 / 46.76 / 46.41
con 3e-4 / 1e-3 / 3e-3); E11 lr 3e-3 (47.13 / 47.05 / 48.62 / 42.97 con 3e-4 / 1e-3 / 3e-3 / 1e-2); pruebas sin
entrenar, con E1u + E10 y con E1u + E11: umbral 0.9, mezcla con tope 0.5 (ventana 7 y 15; otra vez, la ventana
casi no importa) y relleno con a = 0.5.

| Modelo | completa | cruda | aleatoria 10 / 30 / 50 % | regiones | sin | ruido 10 / 20 / 50 cm |
|---|---|---|---|---|---|---|
| E1u (referencia) | 47.31 ± 0.10 | igual | igual | igual | igual | igual |
| E2u (referencia) | 48.59 ± 0.15 | 47.71 | 47.89 / 44.94 / 43.90 | 47.72 | 36.84 ± 0.70 | 47.88 / 47.12 / 45.47 |
| E10 | 49.24 ± 0.24 | 48.10 | 48.39 / 45.04 / 43.13 | 48.17 | 39.15 ± 1.43 | 48.13 / 47.19 / 45.25 |
| E11 | 51.06 ± 0.16 | 47.71 | 48.48 / 39.46 / 33.39 | 49.19 | 15.75 ± 1.21 | 47.73 / 45.40 / 39.54 |
| E1u + E1u (control) | 48.50 ± 0.08 | igual | igual | igual | igual | igual |
| Mezcla E1u/E2u (referencia) | 49.70 ± 0.07 | 49.24 | 49.31 / 48.08 / 47.82 | 49.41 | 47.27 ± 0.10 | 49.22 / 48.71 / 47.61 |
| Interruptor E1u/E10 | 49.17 ± 0.24 | 47.27 | 47.72 / 47.27 / 47.27 | 47.78 | 47.27 ± 0.10 | 48.07 / 47.14 / 45.22 |
| Mezcla E1u/E10 | 49.36 ± 0.15 | 48.87 | 48.90 / 47.74 / 47.45 | 48.90 | 47.27 ± 0.10 | 48.91 / 48.45 / 47.54 |
| Relleno + E1u/E10 | 49.36 ± 0.15 | 49.19 | 49.37 / 49.39 / 49.39 | 49.15 | 45.98 ± 0.19 | 48.91 / 48.45 / 47.54 |
| Interruptor E1u/E11 | 50.98 ± 0.17 | 47.27 | 47.65 / 47.27 / 47.27 | 48.57 | 47.27 ± 0.10 | 47.66 / 45.35 / 39.52 |
| Mezcla E1u/E11 | 52.12 ± 0.25 | 50.15 | 50.52 / 48.16 / 47.56 | 50.96 | 47.27 ± 0.10 | 50.46 / 49.39 / 47.04 |
| Relleno + E1u/E11 | 52.12 ± 0.25 | 51.69 | 52.13 / 52.14 / 52.12 | 51.88 | 42.91 ± 0.42 | 50.46 / 49.39 / 47.04 |

Las pruebas sin entrenar promedian probabilidades: ahí E10 da 49.17 y E11 50.98 (con logits, 49.24 y 51.06). La
desviación entre semillas no pasa de 1.43 (E10 sin profundidad).

**Diagnósticos** (exploratorios, sobre el test; `src/diagnosticos_d05.py`; no sirven para elegir modelos):

- **Con los rasgos corregidos, las tres escalas llevan información (E10).** Con la profundidad de otra imagen en
  una sola escala, E10 pierde 0.42 / 1.41 / 3.15 en 1/8 / 1/16 / 1/32 (media de las 3 semillas), y 8.68 en las
  tres (E2u, en D-04: 0 / 8.12 / 0). Las sumas ya no son grandes frente al color: 0.42 / 0.61 / 0.27 (E2u: 9.3 /
  1.53 / 2.79).
- **E10 suma poco sobre E2u.** +0.65 con profundidad completa (+0.47, +0.67 y +0.81 por semilla). Mejora en 32
  de 40 clases y recupera las que E2u había perdido frente a E1u: cómoda 50.8 → 56.3 (E1u 55.6) y refrigerador
  59.6 → 64.5 (E1u 63.8). La pizarra cae 8.6 (inestable, D-04).
- **La profundidad dentro del codificador es lo que más suma (E11).** E11 − E10 = +1.82 (+1.69 a +1.96 por
  semilla) con 16 k parámetros más, y E11 supera a E10 en la validación con las tres lr comunes (+1.68, +0.29 y
  +2.21 con 3e-4, 1e-3 y 3e-3). Sobre E1u, +3.75. Por clase, E11 − E10 es positivo en 33 de 40 (televisor +7.8,
  cómoda +7.4, mesa de noche +6.5, bañera, cama y otros muebles +4.2, mesa +4.1, espejo +3.9) y no se parece a la
  ganancia de la fusión en la cabeza (correlación con E2u − E1u: 0.01).
- **En E11, la profundidad se usa sobre todo dentro del codificador.** Con la profundidad de otra imagen solo a la
  entrada de la última etapa del color, E11 cae de 51.06 a 31.09; solo en las tres sumas de la cabeza, a 47.01;
  en las dos, a 21.81. W'·P4 mide 0.89 del rasgo de color de 1/16 al que se suma.
- **El costo: E11 necesita una profundidad limpia.** Con la profundidad cruda del sensor (huecos reales) cae a
  47.71, lo mismo que E2u y menos que E10 (48.10); con ruido de 10 cm, a 47.73; sin profundidad, a 15.75. La
  caída sin profundidad viene sobre todo del codificador: con la profundidad vacía solo ahí, 25.71; solo en la
  cabeza, 42.74. E11, como E2u y E10, se entrenó siempre con la profundidad rellenada.
- **Con E1u de respaldo.** La mezcla E1u/E11 da 52.12 ± 0.25 con profundidad completa: +3.62 sobre el control y
  +2.42 sobre E1u/E2u; sin profundidad es E1u. Con la cruda da 50.15 (E1u/E2u: 49.24). El relleno recupera los
  huecos (cruda 51.69, aleatoria 52.1, regiones 51.88), pero sin profundidad no hay de dónde rellenar (42.91).
  Ninguna variante protege del ruido, que la máscara no ve: con 50 cm, 47.04, 0.23 por debajo de E1u.
- **Por qué E1u/E10 suma tan poco.** Con profundidad completa, promediar con E1u añade a E10 solo 0.19 (49.17 →
  49.36), frente a 1.18 en E2u y 1.14 en E11. `combinar.py` empareja modelos de la misma semilla, que ven las
  imágenes en el mismo orden y con el mismo aumento, y E10 usa la misma lr que E1u (1e-3): su última etapa se
  aleja de ADE20K lo mismo (0.053 frente a 0.055; con 3e-3, 0.105 en E2u y 0.104 en E11), y E10 coincide con su
  E1u en el 86.6 % de los píxeles, más que dos E1u de semillas distintas entre sí (85.7 %; E2u 83.6 %, E11 80.9 %).
  Con el E1u de la semilla siguiente, el promedio 0.5 / 0.5 da 50.07 con E10, 50.00 con E2u y 52.24 con E11 (con
  la misma semilla: 49.36, 49.71 y 52.12). Las cifras de la tabla siguen el protocolo fijado; la de E10 se queda
  corta frente a lo que da su mezcla.

Siguiente paso: por decidir (D-06).

## D-06 · Auditoría de la v3: rasgos corregidos, E3 y el factorial 2×2 de E5

**Fecha** 2026-10-29 · **Estado** 🟡 provisional, pendiente de aceptación del equipo · **Origen** pedido del equipo
(Mónica Abarca): auditoría completa de la v3 antes de seguir gastando GPU. Se registra **antes** de entrenar
nada de lo que propone. Detalle y evidencia: `auditoria/` (01 a 07).

- **Hallazgos que cambian la interpretación** (sin modificar ningún resultado guardado):
  - P3 y P5 de `dense_bifpn` no dependen de la entrada (D-05). Se confirma que la ruta de extracción es la
    oficial y que, con pesos no degenerados, la misma ruta da P3 y P5 variables. Falta medir las normas con los
    pesos oficiales (`tools/verificar_defm.py --nyuv2`). Los benchmarks oficiales de DeFM solo usan P4.
  - Con claves y valores constantes, MLCA suma un **sesgo** que solo desaparece cuando ninguna celda es válida: es
    el mecanismo del colapso de E4 (0.78) en 1/8 y 1/32. **E4 de la v3 no permite evaluar la hipótesis.**
  - Sin dropout, la máscara no actúa al entrenar (la profundidad rellenada da m = 1): E3 y E4 se entrenan igual
    con la misma semilla, y la máscara es una intervención solo en la inferencia.
  - La v3 usaba una máscara fraccional; el resumen y el póster dicen binaria con umbral.
  - E5 de la v3 cambiaba el dropout y, a la vez, el orden de los datos (generador compartido).
  - «aleatoria» casi no activa la máscara fraccional (ninguna celda queda vacía); la máscara actúa en «cruda»,
    «regiones» y «sin».
  - Todo el test de la v3 es exploratorio (usado en D-03 a D-05).
- **Decisión.**
  - Familia nueva «c», con los rasgos `c3p4c5` de D-05 y el codificador de color congelado como E1-E5: **E2c**,
    **E3c** (sin máscara), **E4c** (máscara **binaria**, umbral 0.5 fijado ahora; otro valor solo en la
    validación), **E3cd** y **E4cd** (+ dropout 0.3, la tasa declarada), y **E4cf/E4cfd** (máscara fraccional,
    ablación aparte). Dentro de cada par solo cambia la máscara. El dropout usa un generador propio. La lr se
    elige en la validación con la regla del extremo. Los IDs históricos (E1-E11) no cambian de configuración.
  - E3c y E4cf se obtienen **sin entrenar** a partir de E4c, y E4cfd a partir de E4cd (`--derivar`), porque
    tienen la misma trayectoria. Se comprueba con un E3c entrenado (semilla 0).
  - Condiciones nuevas solo para D-06: `regiones_025`, `regiones_075`, `regiones_100`.
  - Ingeniería: DeFM en un commit fijo y SHA-256 de los pesos; huella de configuración y ConfigConflict;
    checkpoints reanudables; errores propagados; `por_imagen.npz`; combinaciones con logits ampliados (las de la
    v3 no se tocan); desempate determinista de la lr; 93 pruebas.
  - Orden: `tools/cola_d06.sh` (`auditoria/06_plan_experimentos.md`).
- **Comprobado antes de lanzar** (CPU, datos sintéticos, pesos aleatorios): 91 pruebas aprobadas y 2 de GPU
  saltadas (`auditoria/evidencia/pytest_cpu.txt`). Las tablas recalculadas desde los JSON reproducen las cifras
  publicadas. **Pendiente en la máquina del equipo:** `reproducir_historicos.py` (que E1u y E2u den sus cifras
  guardadas con el código nuevo) y la verificación de DeFM con los pesos oficiales.
- **Medida y criterio** (propuestos; el equipo los acepta o cambia antes de abrir resultados): diferencia pareada
  E4c − E3c en cruda, regiones (0.5) y sin, con el IC por remuestreo de imágenes; «aporta» si es > 0, el IC
  excluye el 0 y el signo se repite en las 3 semillas (Holm para las tres condiciones). Ver `06_plan`.
- **Predicciones del equipo:** (pendientes; escribir aquí antes de ver ningún resultado de E2c, E3c, E4c, E3cd o
  E4cd).
- **Predicciones de la IA** (Claude, 2026-10-30, antes de lanzar la cola): `predicciones/D06_claude.md`
  (commit b11ecf9). Van aparte y no sustituyen las del equipo.
- **IA.** Hizo la auditoría, encontró A2-A5, A8, A14 y A19 (`auditoria/01`), propuso los diseños, los umbrales y
  el criterio, y escribió el código, las pruebas y los documentos. No se cambió ni se reinterpretó ningún
  resultado guardado: las cifras de la v3 siguen en `runs/rgbd/` con sus identificadores originales.

## Resultados de D-06 (familia «c», factorial 2×2 y control de la derivación)

Medido del 2026-10-30 al 2026-11-01 (test, 3 semillas; `tools/cola_d06.sh`, reanudada tras un corte de energía
sin repetir nada). **Predicciones del equipo:** no se escribieron; Joaquín decidió el 2026-11-01 abrir los
resultados sin ellas. Las de la IA (`predicciones/D06_claude.md`) se contrastan en
`predicciones/D06_claude_resultado.md`. Elegido en la validación por escenas: E2c 3e-3; E4c (y E3c, E4cf) 3e-3;
E3cd 1e-3; E4cd (y E4cfd) 3e-4, tras probar 1e-4 por la regla del extremo (43.01 frente a 43.36).

| Modelo | completa | cruda | aleatoria 10 / 30 / 50 % | regiones 0.25 / 0.5 / 0.75 / 1.0 | sin | ruido 10 / 20 / 50 cm |
|---|---|---|---|---|---|---|
| E1 (referencia) | 44.99 ± 0.30 | igual | igual | igual | igual | igual |
| E2 (referencia) | 47.31 ± 0.26 | 46.97 | 46.66 / 44.82 / 43.90 | — / 46.58 / — / — | 42.25 ± 0.45 | 46.66 / 46.25 / 45.25 |
| E2c | 47.53 ± 0.17 | 43.32 | 44.99 / 35.79 / 30.38 | 46.64 / 44.79 / 41.41 / 37.29 | 20.33 ± 2.48 | 44.61 / 42.96 / 39.26 |
| E3c (derivado) | 44.99 ± 0.20 | 41.19 | 40.37 / 30.65 / 22.90 | 44.42 / 43.15 / 40.44 / 37.46 | 13.71 ± 5.13 | 42.17 / 40.60 / 35.82 |
| E4c | 44.99 ± 0.20 | 40.56 | 40.37 / 30.65 / 23.36 | 44.53 / 43.77 / 42.06 / 39.74 | 14.86 ± 4.18 | 42.17 / 40.60 / 35.82 |
| E4cf (derivado) | 44.99 ± 0.20 | 40.44 | 40.38 / 30.71 / 22.97 | 44.56 / 43.80 / 42.04 / 39.71 | 14.86 ± 4.18 | 42.17 / 40.60 / 35.82 |
| E3cd | 45.45 ± 0.29 | 44.91 | 45.21 / 44.03 / 42.15 | 45.39 / 45.15 / 44.63 / 44.05 | 45.31 ± 0.25 | 45.03 / 44.63 / 43.61 |
| E4cd | 45.21 ± 0.33 | 44.91 | 44.88 / 43.27 / 42.23 | 45.22 / 44.93 / 44.40 / 43.64 | 46.16 ± 0.26 | 44.88 / 44.63 / 43.80 |
| E4cfd (derivado) | 45.21 ± 0.33 | 44.93 | 44.88 / 43.28 / 42.28 | 45.22 / 44.96 / 44.42 / 43.66 | 46.16 ± 0.26 | 44.88 / 44.63 / 43.80 |

**Criterio** (`tools/criterio_d06.py`, Holm sobre cruda, regiones y sin; salida en `runs/rgbd/criterio_d06.txt`):

| Contraste | cruda | regiones | sin |
|---|---|---|---|
| E4c − E3c (máscara, sin dropout) | −0.63, **perjudica** | +0.63, **aporta** | +1.15, no concluyente (semillas −1.13 / +10.67 / −6.10) |
| E4cd − E3cd (máscara, con dropout) | +0.01, no aporta | −0.21, no aporta | +0.85, **aporta** (+0.80 / +0.81 / +0.93) |
| E3cd − E3c (dropout, sin máscara) | +3.71, aporta | +2.00, aporta | +31.60, aporta |
| E4cd − E4c (dropout, con máscara) | +4.35, aporta | +1.16, aporta | +31.30, aporta |
| E3c − E2c (atención frente a suma) | −2.12, perjudica | −1.65, perjudica | −6.62, perjudica |

- **MLCA en la cabeza no aprovecha la profundidad.** Con profundidad completa, E3c = E4c = 44.99, lo mismo que
  E1, y 2.54 bajo E2c (E3c − E2c = −2.54 en completa). Con dropout, 45.21-45.45 (+0.2-0.5 sobre E1).
- **La máscara binaria: ayuda con huecos grandes, perjudica con los del sensor.** Sin dropout, +0.63 en regiones
  (+1.61 y +2.28 con 0.75 y 1.0) y −0.63 en cruda. Con dropout, ninguna diferencia salvo «sin» (+0.85); E4cd y
  E3cd eligieron lr distintas (3e-4 y 1e-3), así que ese +0.85 no es solo la máscara.
- **La robustez viene del entrenamiento, no de la máscara** (el criterio de `06_plan`): E3cd ≈ E4cd, y el dropout
  no cuesta nada con profundidad completa (+0.46 y +0.22) y arregla también los huecos parciales (cruda +3.7 y
  +4.4).
- **E2c, más dependiente y más frágil que E2.** Con los rasgos corregidos, la suma gana solo +0.22 sobre E2 con
  profundidad completa, pero con la cruda cae a 43.32 (E2: 46.97) y sin profundidad a 20.33 (E2: 42.25).
- **Control de la derivación.** E3c entrenado de verdad (semilla 0): 44.48 en completa, frente a 45.19 del
  derivado. La profundidad rellenada de entrenamiento no tiene ningún cero (comprobado: 0 de 795 imágenes), así
  que la máscara no actúa al entrenar y la diferencia es el no determinismo de la GPU acumulado en 5940 pasos.
  La derivación vale, pero diferencias menores de ~0.7 entre modelos entrenados por separado y con una semilla son
  ruido.

**Diagnósticos** (exploratorios, validación por escenas, semilla 0; `runs/rgbd/diagnosticos_d06_validacion.json`,
con E2c añadido; la salida de la cola sin E2c queda en `diagnosticos_d06_validacion_cola.json`):

- **Lo que suma MLCA es más constante que lo que suma la suma.** Parte de Δ que varía con la posición (1/8, 1/16,
  1/32): E2c 0.86 / 0.78 / 0.79; E4c 0.41 / 0.56 / 0.29. En 1/32, el Δ de E4c mide 1.35 veces el rasgo de color.
- **De ahí el colapso sin profundidad, con máscara o sin ella.** Quitando Δ solo en 1/32, E4c cae de 43.18 a
  13.76 (en todas, 9.28); sin profundidad, coincide con E1 en el 10.3 % de los píxeles y pone el 92.4 % en tres
  clases. E2c depende de la profundidad, pero repartida: quitando Δ en todas, 46.38 → 37.44.
- **Con dropout, la cabeza deja de usar la fusión.** En E4cd, quitar Δ no cambia nada (43.36 → 43.50) y Δ es
  pequeño (0.16 / 0.30 / 0.46 del rasgo de color): E4cd se comporta como un modelo de solo color, por eso da lo
  mismo con profundidad o sin ella.

**Prueba sin entrenar pendiente de D-05** (2026-10-30, logits, test): `relleno_interruptor` en
`src/combinar.py` (relleno si la imagen tiene al menos `umbral` de profundidad válida; E1u si no; elegido en la
validación: umbral 0.3, a = 0.5). Con E1u + E11: 52.21 ± 0.24 (completa), 51.77 (cruda), 52.22 / 52.23 / 52.21
(aleatoria), 51.96 (regiones), **47.30 ± 0.11 (sin, igual que E1u)**, 50.52 / 49.44 / 47.09 (ruido). Es el mejor
sistema medido: el relleno de D-05 sin su caída sin profundidad (42.91). Control, dos E1u: 48.53 ± 0.08. Son dos
modelos (el doble de cómputo) y el umbral y a se eligieron entre 20 variantes en la validación.

## D-07 · E11 robusto en un solo modelo; la MLCA, a la entrada del codificador

**Fecha** 2026-11-01 · **Estado** 🟡 provisional · **Origen** Joaquín: «la idea es tener los mejores resultados»,
con las predicciones de la IA como registro previo por ahora; decisiones delegadas en la IA.

- **Lectura de D-06 que decide:** la MLCA en la cabeza está cerrada (no gana con profundidad completa y su máscara
  no da robustez; la da el dropout), y la profundidad rinde dentro del codificador (E11: 51.06, +3.75 sobre E1u),
  pero E11 se hunde sin ella (15.75).
- **Decisión, en este orden:**
  1. **E11d** = E11 + dropout 0.3 (generador propio: mismo orden de datos que E11 por semilla). Un solo modelo
     que aspire a lo que hoy da el par E1u + E11. Solo dropout, sin huecos al entrenar: en D-06 el dropout ya
     arregló los huecos parciales. Predicciones: `predicciones/D07_claude.md`.
  2. **E2cd** = E2c + dropout 0.3: la suma con el mismo entrenamiento robusto que E4cd. Sin él, «MLCA + dropout
     frente a suma» compara entrenamientos distintos.
  3. **MLCA a la entrada de la última etapa del color** (donde E11 concatena), con y sin dropout: la versión de
     la hipótesis del curso donde la profundidad rinde más. Requiere código nuevo (`rgbd.py`: hoy el codificador
     solo admite concatenación) y pruebas.
- Cada paso con sus predicciones escritas antes de lanzarlo.
