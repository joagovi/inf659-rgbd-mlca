# Trabajo relacionado: dónde se ubica la MLCA

Resumen de la revisión bibliográfica del equipo (octubre de 2026), con los resultados de este repositorio al lado.
La revisión siguió las citas hacia atrás y hacia adelante desde 13 artículos semilla (unos 2 150 artículos), más un
segundo salto por los que citan a UCTNet y a EfficientViT. Cada cifra lleva su fuente; «preprint» marca lo que no
tiene revisión por pares confirmada. Las cifras de distintas fuentes no siempre son comparables (otras redes, otras
épocas, otro hardware).

## 1. Los trabajos más cercanos

Ninguno reúne las cuatro piezas de la propuesta, pero cada pieza tiene precedentes. El antecedente más cercano es
**UCTNet**: ya usa la máscara binaria del sensor dentro de la atención («zeros denote no sensor reading at this
location and one denotes a valid sensor reading», §3.2), restando ∞ a las claves inválidas antes de la softmax.
Por eso «máscara del sensor dentro de la atención» no es una novedad de este trabajo; lo que cambia es dónde y cómo.

| Trabajo | Máscara del sensor por píxel en la fusión | Atención de costo lineal | Salida o inyección iniciada en cero | Evalúa huecos parciales |
|---|---|---|---|---|
| UCTNet (ECCV 2022) | Sí, en la autoatención de la rama de profundidad | No (softmax, Swin-S) | No | No consta |
| RSGMamba (preprint 2026) | No (fiabilidad aprendida) | Sí (SSM) | No | No |
| ConD (preprint 2026) | No (modalidad entera) | No | Sí (1×1 en cero) | No |
| GeomPrompt (CVPR W 2026) | No (resintetiza la profundidad) | No | Sí | Sí (7 corrupciones) |
| DiffusionMMS (preprint) | No (atención deformable) | No | No | Subconjunto «Invalid» |
| CM-SSM (IROS 2025, RGB-térmico) | No | Sí (SSM), con dos EfficientViT-B1 | No | No |
| Cao et al., BPLCA (preprint 2026) | No | Sí (kernelizada) | No | No |
| **MLCA (este trabajo)** | **Sí, en una atención cruzada (color → profundidad)** | **Sí (ReLU)** | **Sí: sin profundidad, g = f exacto** | **Sí: huecos del sensor, aleatorios, regiones, ruido, sin profundidad** |

Fuentes: UCTNet ([ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900020.pdf)); RSGMamba
([arXiv 2604.12319](https://arxiv.org/abs/2604.12319)); ConD ([arXiv 2607.20326](https://arxiv.org/abs/2607.20326));
GeomPrompt ([arXiv 2604.11585](https://arxiv.org/abs/2604.11585)); DiffusionMMS
([arXiv 2409.15117](https://arxiv.org/abs/2409.15117)); CM-SSM ([arXiv 2506.17869](https://arxiv.org/abs/2506.17869));
Cao et al. ([arXiv 2610.17800](https://arxiv.org/abs/2610.17800)).

**Genealogía.** Con la máscara en el resumen S = Σ m_j φ(k_j) v_jᵀ y en el normalizador z = Σ m_j φ(k_j), la atención
lineal enmascarada equivale a la **convolución normalizada** de Knutsson y Westin (1993), con un núcleo global
aprendido (álgebra propia). De esa idea descienden las CNN invariantes a la dispersión (Uhrig et al., 2017), la
propagación de confianza (Eldesokey et al., 2018), la convolución parcial (Liu et al., 2018) y MAT (CVPR 2022), que
agrega solo desde los tokens válidos. La inicialización en cero viene de ControlNet y LLaMA-Adapter.
Fuentes: [Knutsson y Westin](https://doi.org/10.1109/CVPR.1993.341081); [Uhrig et al.](https://arxiv.org/abs/1708.06500);
[Eldesokey et al.](https://arxiv.org/abs/1811.01791); [convolución parcial](https://arxiv.org/abs/1804.07723);
[MAT](https://arxiv.org/abs/2203.15270); [ControlNet](https://arxiv.org/abs/2302.05543);
[LLaMA-Adapter](https://arxiv.org/abs/2303.16199).

## 2. Resultados publicados en NYUv2 (40 clases, 795/654), junto a los nuestros

**Robustez: entrada completa y sin profundidad (mIoU).**

| Método | Completa | Sin profundidad | Fuente |
|---|---|---|---|
| CMNeXt MiT-B4 | 56.30 | 51.19 | [Reza et al.](https://arxiv.org/abs/2310.03986) |
| Reza et al. (SSF) | 56.30 | 52.82 | ídem |
| DFormer-B | 55.6 | 35.2 | [ConD, Tabla I](https://arxiv.org/abs/2607.20326) |
| DFormer-B + ConD | 55.7 | 48.3 | ídem |
| Sigma-S + ConD | 57.2 | 52.9 | ídem |
| ESANet R50 con *dropout* | 47.23 | 41.34 | [MMANet, Tabla 4](https://arxiv.org/abs/2304.08028) |
| MMANet (ESANet R50) | 49.62 | 44.93 | ídem |
| **E1u (solo color, este trabajo)** | **47.31** | **47.31** | `ARQUITECTURAS.md` |
| **E11d (suma + dropout, este trabajo)** | **51.26** | **47.43** | ídem |
| **E11r (ídem, entrenado con la cruda)** | **50.63** (cruda: 51.39) | **47.31** | ídem |
| **E4ed (MLCA + dropout, este trabajo)** | **49.24** | **47.29** | ídem |

**Precisión y velocidad en una Jetson AGX Orin** (TensorRT 10.3, FP16, 640×480; medido por el grupo de ESANet):

| Modelo (fusión) | FPS | mIoU en NYUv2 |
|---|---|---|
| EMSANet (2×ResNet-34, SE + suma) | 78.19 | 50.97 |
| EMSAFormer (SwinV2-T) | 62.63 | 51.06 |
| DFormer-S | 55.03 | 53.6 |
| DFormer-B | 35.11 | 55.6 |
| CMX 2×MiT-B2 (atención de Shen, costo lineal) | 33.37 | 54.1 |
| CMNeXt 2×MiT-B4 | 14.84 | 56.9 |

Fuente: [DVEFormer, Tabla I](https://arxiv.org/abs/2601.00359). ESANet R34-NBt1D: 50.30 en NYUv2 y 29.7 FPS en una AGX
Xavier ([ESANet](https://arxiv.org/abs/2011.06961)). No medimos en Jetson; en una RTX 4060 Laptop, E11d tarda unos
27 ms (`runs/coste/`).

**Cómo leer la comparación.** Nuestros modelos están a la altura de ESANet y EMSANet con extractores mucho más chicos
y casi todo congelado, y por debajo de los modelos grandes (54-57). La receta estándar en NYUv2 son unas 500 épocas
con GPUs de 24-80 GB; nosotros entrenamos 60 épocas en una GPU de 8 GB. Sin profundidad, E11d pierde 3.8 puntos,
en el rango de los métodos robustos (4-7). Casi ningún trabajo entrena ni evalúa con la profundidad cruda
(`rawDepths`): ESANet tiene la opción, desactivada por defecto ([args.py](https://github.com/TUI-NICR/ESANet/blob/main/src/args.py)).

## 3. Huecos parciales: lo que mide la literatura

Casi toda la literatura modela la ausencia como la pérdida de una modalidad entera. La medida sistemática de huecos
parciales es la de **Liao et al.** (CVPR W 2025), que pone a cero una fracción de la profundidad. En DeLiVER (datos
sintéticos de conducción, MiT-B2), con solo un 25 % de huecos los mejores modelos pierden entre 19 y 23 puntos:

| Modelo | Profundidad intacta | 25 % a cero | 50 % | 75 % | Toda a cero |
|---|---|---|---|---|---|
| CMNeXt | 66.32 | 47.75 | 38.26 | 30.12 | 22.50 |
| GeminiFusion | 66.93 | 44.42 | 29.37 | 22.57 | 15.8 |
| MAGIC | 66.10 | 43.96 | 42.91 | 42.77 | 42.72 |
| StitchFusion | 68.25 | 47.38 | 41.96 | 38.51 | 33.55 |

Fuente: [Liao et al., Tablas 3-6](https://arxiv.org/abs/2503.18445). Nuestras condiciones «aleatoria» (10, 30 y 50 %)
siguen ese protocolo, por píxel; «regiones» sigue la corrupción *hole* de GeomPrompt; «ruido», el ruido gaussiano de
Liao; y «cruda», los huecos reales del Kinect de NYUv2, que la literatura casi no usa.

## 4. Los dos extractores preentrenados

- **EfficientViT-B1** ([Cai et al., ICCV 2023](https://arxiv.org/abs/2205.14756)): 4.8 M de parámetros, 42.8 mIoU en
  ADE20K, Apache-2.0. Sus rasgos congelados son débiles en sondeo lineal ([AM-RADIO, Tabla 7](https://arxiv.org/abs/2312.06709)),
  lo que es coherente con que ajustar su última etapa ayude (E1 44.99 → E1u 47.31).
- **DeFM** ([Patel et al., arXiv 2601.18923](https://arxiv.org/abs/2601.18923)): DINOv2 aplicado a unos 60 M de imágenes
  de profundidad, con un maestro ViT-L/14 y destilados de 3 a 29 M de parámetros, Apache-2.0. En segmentación solo con
  profundidad (sondeo lineal), DeFM ResNet-18 da 19.85 en SUN RGB-D y 22.47 en ScanNet; DeFM ViT-S/14, 27.78 y
  27.69 (Tabla V). No había resultados publicados de DeFM como rama de una red RGB-D. Su preentrenamiento ya incluye
  un «pixel dropout» de hasta el 10 % que simula profundidad faltante (Tabla XII). Su preprocesado oficial pone en 0
  los huecos y calcula un canal min-max por imagen que cuenta esos ceros; en este trabajo el min-max se calcula solo
  con los píxeles medidos (D-01). Entre sus datos de preentrenamiento hay interiores reales como ScanNet y SUN3D, que
  hay que declarar si se evalúa en SUN RGB-D.

## 5. Conjuntos de datos considerados

| Conjunto | Qué es | Para qué serviría |
|---|---|---|
| NYUv2 (el de todo el trabajo) | 1 449 imágenes (795/654), 40 clases, Kinect v1, con `depths` (rellenada) y `rawDepths` (cruda) | Huecos reales y estructurados sin descargar nada más |
| SUN RGB-D | 10 335 imágenes (5 285/5 050), 37 clases, 4 sensores | Cambio de sensor (Fase C, no realizada); incluye las imágenes de NYUv2, que habría que excluir |
| DeLiVER | 7 885 vistas sintéticas (CARLA) | Protocolo de Liao; su profundidad llega como HHA, no métrica |

Fuentes: [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html); [SUN RGB-D](https://rgbd.cs.princeton.edu/paper.pdf);
[DeLiVER](https://github.com/jamycheung/DELIVER).

## 6. Dónde queda la contribución

La idea de que un hueco no vote en la atención no es nueva (convolución normalizada, MAT, UCTNet). Lo que aporta
este trabajo es (1) llevar la máscara a una atención cruzada de costo lineal, donde la exclusión es exacta; (2) la
garantía por construcción de que, sin profundidad, la fusión aporta cero; y (3) una evaluación controlada (operador
× lugar × entrenamiento, 3 semillas, criterio registrado antes) con los huecos reales del sensor. El resultado
principal es negativo para el mecanismo: con profundidad, la suma gana, y la robustez viene del entrenamiento
(*modality dropout*), no de la máscara. Queda como trabajo futuro la comparación directa con una atención softmax
enmascarada como la de UCTNet (E6).

*Resumen preparado con asistencia de IA (Claude, Anthropic) a partir de la revisión del equipo; ver `DECISIONES.md`.*
