# 2 · Matriz de coherencia científica: resumen, póster e implementación

Fuentes: el resumen de WAIMLAp 2026 (`Abstract_MLCA_RGBD_Segmentation.pdf`), el póster (`Poster_MLCA_RGBD_Segmentation_editable-FINAL.pdf`), la propuesta revisada (`proposal/PROPUESTA_RGBD_MLCA.md`), la v3 y la versión corregida (v4).

Claves: ✅ coincide · ⚠️ coincide con una desviación que hay que declarar · ❌ no coincide o no está implementado · ⏳ implementado, sin ejecutar.

| Aspecto | Resumen / póster | v3 | v4 (corregida) | Estado | Qué hacer en el informe |
|---|---|---|---|---|---|
| Pregunta | ¿Excluir la profundidad inválida dentro de una atención cruzada lineal reduce la pérdida de precisión con profundidad ruidosa, incompleta o ausente? | Sin E3, no se puede contestar. E4 frente a E2 mezcla atención, máscara y P3/P5 constantes | El factorial E3c/E4c × dropout la contesta | ⏳ | Contestar solo con E4c − E3c (y con dropout) |
| Extractor RGB | EfficientViT-B1 congelado | Codificador de **EfficientViT-Seg-B1 preentrenado en ADE20K**, congelado en E1-E5; última etapa ajustada en la familia «u» | Igual; la familia «c» lo mantiene congelado | ⚠️ | Declarar ADE20K (segmentación, 150 clases), no ImageNet |
| Extractor de profundidad | DeFM ResNet-18 congelado | Congelado; P3-P5 de la BiFPN: **solo P4 depende de la entrada** | `c3p4c5`: C3 y C5 del ResNet (LN sin parámetros) y P4 | ❌ → ⏳ | Declarar que E2-E9 usaron profundidad solo en 1/16 |
| Alineación y proyección | «spatial alignment and projection to a common width» | Las tres escalas coinciden (1/8, 1/16, 1/32); proyecciones 1×1 dentro de cada fusión | Igual (probado) | ✅ | — |
| Fórmula de MLCA | φ(Q)ᵀΣ m φ(K)V / (φ(Q)ᵀΣ m φ(K) + ε), núcleo de [4] | La misma fórmula; φ = ReLU (en [4], elu+1) | Igual | ⚠️ | Declarar φ = ReLU |
| Máscara | m_j ∈ {0,1}: válido si la fracción válida de la celda supera un umbral | **Fraccional**, m_j ∈ [0, 1] | E4c **binaria** (umbral 0.5, fijado de antemano); E4cf fraccional como ablación | ❌ → ⏳ | Informar E4c como MLCA; E4cf aparte |
| «Cost stays linear» | Las celdas enmascaradas salen de las dos sumas | Salen de las sumas, pero se calculan sobre todas las celdas (con ceros): el coste no baja con la máscara | Igual (prueba de coste lineal) | ✅ | — |
| «Output vanishes when no depth is valid» | Sí | Exacto (`test_sin_profundidad_aporta_cero_exacto`) | Igual | ✅ | Añadir: no garantiza la precisión de E1 (E4: 0.78) |
| Residual hacia RGB | «falls back to an RGB-only prediction» | Exacto en los rasgos; la cabeza no es la de E1 | Igual; `test_sin_profundidad_e4_es_la_ruta_de_color` | ⚠️ | No afirmar «≈ E1» sin el dato |
| Qué se entrena | «Only the projections, MLCA and a lightweight decoder» | La **cabeza preentrenada de ADE20K** (clasificador nuevo) y la fusión; en «u», además la etapa 4 | Igual | ⚠️ | «Cabeza de EfficientViT-Seg preentrenada», no «decoder ligero» desde cero |
| Datos | NYUv2, 1 449, 795/654 | ✅; la validación (15 % de escenas) sale del entrenamiento | Igual | ✅ | — |
| Profundidad de entrada | La máscara se calcula «from the raw depth map» (Fig. 1) | Entrenamiento con `depths` rellenada (máscara = 1); «cruda» solo al evaluar | Igual (consecuencia: A3) | ⚠️ | Declarar que «completa» es rellenada y que la máscara no actúa al entrenar sin dropout |
| E1 | Solo RGB | ✅ | ✅ | ✅ | — |
| E2 | «+ DeFM depth, concatenation» | Suma residual 1×1 (W = 0 al inicio) con P3-P5 | E2 (histórico) y E2c (rasgos corregidos) | ⚠️ | «Concatenación con proyección 1×1 cuyo bloque RGB es la identidad» |
| E3 | Atención cruzada lineal sin máscara | **No existe** | E3c (= E4c con m = 1), derivable de E4c; E3cd entrenado | ❌ → ⏳ | — |
| E4 | + máscara de validez (MLCA) | Fraccional, con P3/P5 constantes | E4c (binaria), con rasgos corregidos | ⚠️ → ⏳ | E4 de la v3 como resultado exploratorio de una configuración defectuosa |
| E5 | E4 con y sin *modality dropout*; «si E5 sin la máscara ≈ E4…» | Solo E4 + dropout; **falta E3 + dropout** | Factorial 2×2: E3c, E3cd, E4c, E4cd | ❌ → ⏳ | El efecto «arquitectura frente a entrenamiento» sale del factorial |
| E6 (propuesta, no en el resumen) | Softmax enmascarada como UCTNet | No | No (fuera de la prioridad) | ❌ | Dejarlo como trabajo futuro |
| Condiciones | Completa, 10/30/50 %, contiguas, ruido gaussiano, ausencia total | ✅ y «cruda»; ruido D-02 provisional | Más severidades de «regiones» (solo D-06) | ✅ / ⚠️ | Declarar que «aleatoria» casi no activa la máscara (A19) |
| Métricas | mIoU, IoU por clase, caída, params, FLOPs, latencia, VRAM; media ± DE de 3 semillas | mIoU, IoU por clase, 3 semillas; **sin coste de la red completa** (solo el bloque, en `09`) | `estadistica.py` (IC, pares, bootstrap) y `coste.py` (params, FLOPs, latencia, VRAM) | ⚠️ → ⏳ | Coste con `coste.py` en la GPU |
| Hipótesis E2 vs E1 | Sube con profundidad completa y baja con huecos | Se cumple (+2.32; −1.09 con 50 %; −2.74 sin) | — | ✅ (exploratorio) | — |
| Hipótesis E3 vs E2 | Mejor fusión, sensibilidad parecida | No medible | Pendiente (E3c frente a E2c) | ⏳ | — |
| Hipótesis E4 vs E3 (principal) | Menor caída con huecos; ≈ E1 sin profundidad; nada con ruido | No medible; E4 colapsa sin profundidad | Pendiente (E4c − E3c, pareado) | ⏳ | Preregistrar antes de evaluar (06_plan) |
| Hipótesis E5 | El dropout solo puede igualar a E4 | E5 no colapsa (45.74), pero también cambia el orden de los datos (A5) | Pendiente (E3cd frente a E4cd) | ⏳ | — |
| Coste | «small computational cost», «bounded increase» | Sin medir de extremo a extremo | En CPU: DeFM añade 24.3 GFLOPs a 7.2 de E1 (×4.4); MLCA añade 0.4-0.6 GFLOPs sobre la suma | ⚠️ | El coste lo domina DeFM, no la atención; decirlo |
| Resultado principal del README/LEEME | — | La mejor cifra (49.70) es la mezcla E1u + E2u, sin MLCA | Se mantiene separada (familia complementaria) | ⚠️ | No presentarla como resultado de MLCA |
| Novedad / limitaciones | Incremental; máscara binaria; NYUv2 pequeño; extractores congelados; fallos simulados | Coherente | Añadir: P3/P5 constantes de DeFM; test ya usado para decidir | ⚠️ | Añadir a las limitaciones |

## Lo que el póster puede decir hoy sin nuevas corridas

- E1 y E2 (exploratorios, 3 semillas) y que la profundidad sin máscara degrada con huecos.
- Que la configuración de E4 en la v3 no permite evaluar la hipótesis (A1-A3), con el colapso explicado.
- El diseño corregido (factorial 2×2) y el coste medido: parámetros y FLOPs (válidos en cualquier máquina); la latencia, solo en la GPU.

No debe decir que MLCA mejora la robustez, ni presentar 49.70 como resultado de MLCA.
