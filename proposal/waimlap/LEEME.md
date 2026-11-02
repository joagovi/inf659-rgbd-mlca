# Resumen extendido y póster de WAIMLAp 2026

| Archivo | Qué es |
|---|---|
| `Abstract_MLCA_RGBD_v2.docx` y `.pdf`, `Poster_MLCA_RGBD_v2.pptx` y `.pdf` | Borradores v2 (22 de octubre de 2026), con E6 (softmax enmascarada) |
| `Abstract_MLCA_RGBD_Segmentation.pdf` | Resumen extendido del equipo (2 páginas, todavía sin resultados), con las expectativas escritas antes de entrenar |
| `Poster_MLCA_RGBD_Segmentation.pdf` | Póster del equipo («work in progress», 25 de octubre de 2026) |

## Qué hay que cambiar para la versión final

| Dicen el resumen o el póster | Lo que pasó (dónde está) |
|---|---|
| Los dos codificadores están congelados | DeFM, sí. Del color, la última etapa se ajusta en los mejores modelos (E1u, E10, E11, E11d, E11r), y en ellos la profundidad entra también al codificador (`ARQUITECTURAS.md`) |
| Sin profundidad, la vía residual devuelve la predicción de solo color | Solo con *modality dropout*. Sin él, la MLCA en la cabeza cae a 14.86 sin profundidad (E4c), frente a 44.99 de solo color (E1) |
| CMX tiene costo cuadrático | CMX ya usa una atención cruzada eficiente, de costo lineal (Shen et al.), como dice la propuesta (`proposal/PROPUESTA_RGBD_MLCA.md`). Lo nuevo es la máscara de validez dentro de ella |
| E2 es una concatenación | Se implementó como f + W·d (W, una 1×1 iniciada en cero): equivale a concatenar y aplicar una 1×1 con el bloque del color fijo en la identidad. Conviene un solo nombre: «suma» |
| Los rasgos de DeFM | Se usan C3 y C5 del ResNet con P4 de la BiFPN, porque en los pesos publicados P3 y P5 son constantes (D-05); el canal min-max se calcula solo con los píxeles válidos (D-01). DeFM sigue congelado |
| Todavía no hay resultados | Los de D-06 a D-08 (`README.md`, `notebooks/12_resultados.ipynb`) y la figura de robustez (`proposal/figs/robustez_d06_d07_en.png`) |

## Las expectativas del resumen, frente a lo medido

El diseño del resumen (fusión en la cabeza, codificadores congelados) es el de D-06. Test de NYUv2, media de 3
semillas.

| Expectativa del resumen | Resultado | |
|---|---|---|
| E2 frente a E1: la profundidad ayuda con la profundidad completa y empeora con huecos | 47.53 frente a 44.99; sin profundidad, 20.33 | ✔ |
| E3 frente a E2: la atención fusiona mejor | E3c 44.99 frente a E2c 47.53 | ✘ |
| E4 frente a E3 (la hipótesis principal): la máscara reduce la caída con huecos, y sin profundidad queda cerca de E1 | con los huecos del sensor, 40.56 frente a 41.19; sin profundidad, los dos colapsan (14.86 y 13.71) | ✘ |
| E5: el dropout solo podría dar la misma robustez, y eso sería un resultado negativo para la MLCA | con dropout, con máscara y sin ella dan casi lo mismo (42.23 y 42.15 con el 50 % de huecos) | ✔ |

El resumen también decía qué haría cambiar de opinión al equipo: que E4 ≈ E3 con huecos (la máscara no aporta) y que
la variante sin máscara con dropout ≈ E4 (la robustez viene del entrenamiento). Pasaron las dos cosas, y D-07 y
D-08 lo confirman con la fusión también en el codificador.
