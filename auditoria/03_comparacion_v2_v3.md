# 3 · v2 frente a v3: qué cambió y qué no se puede comparar

**Limitación de esta comparación.** El código de la v2 (`inf659-rgbd-mlca.zip` del 26/10/2026, la versión
corregida tras la auditoría de E1/E2, con `src/models/encoders.py`, `src/nb.py` y `tests/test_e1_e2.py`) **no
estaba en esta sesión**: está en el Drive del equipo, pero no se pudo traer al espacio de trabajo. Lo que sigue
sobre la v2 sale de su registro de estado (`claude/codigo-mlca-estado.md` en el proyecto), de lo que la v3 dice
de ella (`LEEME_EQUIPO.md`: «la v2 también lee P3–P5») y de la lista de diferencias del encargo. Cada fila de la
v2 marcada «por confirmar» debe verificarse contra su código antes de citarla. La v2 no se ha modificado ni
sobrescrito: este repositorio parte de la v3, y la v3 queda intacta en el primer commit (`git show HEAD~1`).

| Factor | v2 | v3 | ¿Cambia el mIoU por sí solo? |
|---|---|---|---|
| Codificador RGB | EfficientViT-B1 de **clasificación, ImageNet** (por confirmar) | **EfficientViT-Seg-B1 entrenado para segmentación en ADE20K** (150 clases, muchas de interior) | Sí, mucho: la v3 parte de rasgos ya entrenados para segmentar |
| Decoder | MLP entrenado desde cero (por confirmar) | Cabeza de segmentación de ADE20K preentrenada, con el clasificador cambiado a 40 | Sí |
| Rasgos de DeFM | P3-P5 de `dense_bifpn` (confirmado por la v3) | P3-P5 en E2-E9; C3/P4/C5 en E10, E11 y la familia «c» | Sí: P3 y P5 constantes en las dos (A1) |
| Profundidad: `rawDepths` / `depths` | Por confirmar cuál se usó al entrenar y cuál al evaluar | Entrenamiento con `depths` (rellenada); evaluación con las dos («completa», «cruda») | Sí: cambia la fracción válida (30.7 % de huecos en la imagen completa) |
| Preprocesado de DeFM | Oficial (`convert_to_3channel_metric`: mínimo y máximo con los ceros) (por confirmar) | Mínimo y máximo **solo sobre los válidos** (D-01) | Sí, con huecos (desplazamiento mediano de 0.21 en el tercer canal de NYUv2); igual sin huecos |
| Máscara | Binaria (por confirmar el umbral) | Fraccional | Sí, con huecos (A4) |
| Núcleo de la atención | «Original» (por confirmar: elu+1 de [4] o ReLU) | ReLU | Posible (A21) |
| Control de profundidad | C3 = E2 con la profundidad barajada (D22) | Permutación de la profundidad por escala, como diagnóstico (D-04) | No (son diagnósticos) |
| Protocolo de entrenamiento | Pasos con micro-lotes y pérdida normalizada por el lote efectivo (A6 de la v2); reanudación determinista (D19) | AdamW, lote 8, 60 épocas, 100 pasos de calentamiento, caída polinómica, bf16; sin reanudación (v3) | Sí |
| Selección de lr | Por confirmar | Validación por escenas (15 %), 3 lr; regla del extremo solo en E2u/E10/E11 | Sí |
| Evaluación | fp16/bf16 corregido en la v2 (A8) | bf16, logits ampliados a 480×640 | Pequeño |

## Consecuencia

Entre la v2 y la v3 cambian al menos **siete factores a la vez**: preentrenamiento del codificador, cabeza,
preprocesado, máscara, núcleo, protocolo y selección de lr, y quizá la profundidad de entrenamiento. Por eso:

1. **No se puede atribuir ninguna diferencia de mIoU entre v2 y v3 al decoder**, ni a ningún otro factor por
   separado. La diferencia más probable, que no está medida, viene de usar un modelo ya entrenado para segmentar
   (ADE20K) en lugar de un clasificador de ImageNet.
2. Lo que **sí** se transfiere: el hallazgo A1 (P3 y P5 constantes) afecta igual a las dos versiones. Las cifras
   de fusión multiescala de la v2 solo tenían profundidad informativa en 1/16.
3. Para comparar v2 y v3 haría falta una escalera de un factor por paso (v2 → v2 con la cabeza de ADE20K → …). No
   está entre las prioridades del curso (`06_plan_experimentos.md`, prioridad baja).

## Trazabilidad

`v2 (Drive, 26/10, sin cambios)` → `v3 (inf659-rgbd-mlca_v3.zip, f50e012, primer commit de este repositorio)` →
`v4 (segundo commit: correcciones de D-06)`. `git diff HEAD~1` muestra cada cambio sobre la v3.
