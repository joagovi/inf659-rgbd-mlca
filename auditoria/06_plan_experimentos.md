# 6 · Plan priorizado de experimentos pendientes

Objetivo: contestar la pregunta del resumen («¿la máscara dentro de la atención lineal reduce la pérdida con
profundidad incompleta, frente a la misma atención sin máscara?») con el menor gasto de GPU y sin repetir
corridas que no aporten. Tiempos para la RTX 4060 Laptop (8 GB), unos 17 minutos por corrida y unos 3 minutos por
evaluación completa del test. Todo está en `tools/cola_d06.sh` (`bash tools/cola_d06.sh [paso]`), que se detiene
ante cualquier error y reanuda lo cortado.

## Prioridad y coste

| Paso | Qué | Corridas de entrenamiento | GPU aprox. | Por qué |
|---|---|---|---|---|
| 0 | Pruebas, `verificar_defm.py --nyuv2`, `reproducir_historicos.py`, `huecos_efectivos.py --nyuv2` | 0 | 20-30 min | Confirma D-05 con los pesos oficiales, que la v4 reproduce la v3 (E1u, E2u) y que `depths` no tiene ceros (condición de A3) |
| 1 | **E4c** (barrido + 3 semillas) y, **sin entrenar**, E3c y E4cf | 6-8 | 1.7-2.3 h | Celda «sin dropout» del factorial: E4c − E3c con **los mismos pesos** (diferencia pareada pura de la máscara) |
| 2 | **E4cd y E3cd** (barrido + 3 semillas cada uno) y, sin entrenar, E4cfd | 12-16 | 3.4-4.5 h | Celda «con dropout»: aquí la máscara sí cambia el entrenamiento |
| 3 | Control: E3c entrenado de verdad, semilla 0 | 1 | 17 min | Comprueba en la GPU que la derivación del paso 1 vale (debe coincidir salvo el no determinismo de cuDNN) |
| 4 | **E2c** (barrido + 3 semillas) | 6-8 | 1.7-2.3 h | E2 con los mismos rasgos que E3c/E4c: contesta «E3 frente a E2» |
| 5 | Tablas, coste en la GPU (`coste.py`), diagnósticos en la validación | 0 | 30 min | Tablas estadísticas, coste de E1-E5 y diagnósticos sin gastar el test |
| | **Total** | **25-33** | **unas 8-10 h** | |

**No hace falta repetir:** E1 (no usa DeFM; sirve de referencia de color para la familia «c»), E1u, E2u, E7, E8,
E9, E2h ni las combinaciones. E4 y E5 de la v3 se conservan como resultados exploratorios de la configuración con
P3/P5 constantes. Si el equipo conserva los pesos de E4, `python tools/reproducir_historicos.py --sin-mascara` da,
sin entrenar, el «E3 de la v3» (los mismos pesos sin máscara), exploratorio.

**Opcional, en este orden y solo si hay tiempo:**

1. **E4r / E3r: entrenar con `rawDepths`** (la máscara actúa al entrenar también sin dropout). Es la versión más
   fiel a la Fig. 1 del resumen («the validity mask is computed from the raw depth map»). Cambia un solo factor
   respecto de E4c/E3c (la profundidad de entrenamiento) y debe registrarse en un D-07 antes de entrenar.
   Unas 12-16 corridas.
2. **Evaluación independiente en SUN RGB-D, sin reentrenar** (ver abajo). Coste de datos, no de GPU: unos 30 min
   de evaluación.
3. E10/E11 (`tools/cola_d05.sh`): familia «ajuste u», no contestan la hipótesis de MLCA.
4. Ablación del núcleo (elu+1 de [4] frente a ReLU), solo si los diagnósticos muestran muchas consultas anuladas
   (A21).
5. La escalera v2 → v3 y E6 (softmax enmascarada): trabajo futuro.

## Hipótesis a registrar en D-06 antes de entrenar (borrador para el equipo)

Las predicciones son del equipo. La IA propone la forma de medirlas y los umbrales, que el equipo debe aceptar o
cambiar **antes** de abrir cualquier resultado de los pasos 1-4.

- **Medida principal:** diferencia pareada E4c − E3c en mIoU (y en caída respecto a la profundidad completa) en las
  condiciones donde la máscara actúa: **cruda, regiones (0.5) y sin**. Pareada por checkpoint (paso 1) o por
  semilla (paso 2), con el IC 95 % por remuestreo de imágenes (`estadistica.py`) y el signo en las tres semillas.
- **Criterio, escrito antes:** «la máscara aporta» en una condición si la diferencia es > 0, su IC 95 % por
  imágenes excluye el 0 y el signo se repite en las 3 semillas; «no aporta» si |diferencia| < 0.3 y el IC
  incluye el 0; en otro caso, «no concluyente». Corrección de Holm para las tres condiciones principales.
- **Secundarias:** severidades de regiones (0.25, 0.75, 1.0); aleatoria 10/30/50 % (la máscara apenas actúa: se
  espera ≈ 0, A19); ruido (se espera ≈ 0: la máscara no lo ve).
- **Factorial (paso 2):** efecto de la máscara con dropout (E4cd − E3cd), efecto del dropout sin máscara
  (E3cd − E3c) e interacción. «La robustez viene del entrenamiento» si E3cd ≈ E4cd con el mismo criterio.
- **E3 frente a E2 (paso 4):** E3c − E2c con profundidad completa y con huecos.

## Confirmatorio frente a exploratorio

- **Exploratorio:** todo el test de la v3 (D-03 a D-05 lo usaron para diagnósticos y decisiones) y todo
  diagnóstico (`diagnosticos_*.py`).
- **Semiconfirmatorio:** los pasos 1-4 en el test de NYUv2, si las hipótesis de arriba se registran antes y el
  test se evalúa una vez por configuración. No es plenamente confirmatorio, porque la elección de los rasgos
  `c3p4c5` se apoyó en diagnósticos sobre ese mismo test (D-04, D-05).
- **Confirmación independiente (opcional, sin reentrenar):** **SUN RGB-D**, partición de test, **sin** las
  imágenes que vienen de NYUv2. Sus 37 clases son las 37 primeras de NYU40, así que los modelos se evalúan tal
  cual, tomando el argmax sobre las 37 primeras salidas. Tiene profundidad cruda de otros sensores (Kinect v2,
  RealSense, Xtion), con huecos reales. Basta con un lector (`src/sunrgbd.py`, no incluido) y con
  `entrenar_rgbd.evaluar`. Para el curso es un complemento: si no hay tiempo, se declara como limitación.

## Qué cambiaría la conclusión

- E4c ≈ E3c en cruda, regiones y sin → la máscara no aporta (resultado negativo, publicable como tal).
- E3cd ≈ E4cd y ambos ≫ E3c sin profundidad → la robustez viene del dropout, no del operador.
- E4c colapsa sin profundidad también con C3/P4/C5 → el colapso no venía solo de P3/P5. Entonces hay que estudiar
  la dependencia de la cabeza (diagnóstico de escalas), no cambiar la arquitectura sin un experimento propio.
