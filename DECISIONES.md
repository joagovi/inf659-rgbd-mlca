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