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