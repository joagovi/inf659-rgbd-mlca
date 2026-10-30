# Auditoría D-06 · índice y resumen

Auditoría de `inf659-rgbd-mlca_v3` (2026-10-29) y versión corregida. Ningún resultado guardado se modificó ni se
reinterpretó en su archivo: las cifras de la v3 siguen en `runs/rgbd/` con sus identificadores originales, y
`git diff HEAD~1` muestra todos los cambios sobre la v3.

| Archivo | Contenido |
|---|---|
| `01_auditoria_tecnica.md` | 24 hallazgos con gravedad, estado (confirmado, parcial o hipótesis), archivo y líneas, evidencia y corrección; comprobaciones de E1/E2 |
| `02_matriz_coherencia.md` | Resumen y póster frente a v3 y v4, aspecto por aspecto, y qué puede decir hoy el póster |
| `03_comparacion_v2_v3.md` | Los factores que cambian entre v2 y v3 y por qué sus mIoU no se comparan |
| `04_informe_defm.md` | Verificación de D-05: procedencia, código oficial, controles, evidencia con datos y lo pendiente |
| `05_tablas_estadisticas.md` | Generado por `src/estadistica.py` desde los JSON: media ± DE, IC, caídas, pares, por clase, barridos y combinaciones |
| `06_plan_experimentos.md` | Plan priorizado de GPU (8-10 h), hipótesis para registrar, confirmatorio y exploratorio |
| `07_verificacion.md` | Qué se ejecutó de verdad (91 pruebas en CPU) y qué queda pendiente (12 puntos) |
| `evidencia/` | Salida de pytest (texto y JUnit XML) |

## En cinco puntos

1. **E4 de la v3 no prueba la hipótesis.** P3 y P5 de DeFM no dependen de la entrada. Con K y V constantes, MLCA
   suma un sesgo que la máscara solo quita cuando no queda ninguna celda válida: de ahí el colapso a 0.78. E2 no
   colapsa porque su constante se queda.
2. **Sin dropout, la máscara no actúa al entrenar** (la profundidad rellenada da m = 1). E3 y E4 se entrenan igual,
   y la comparación limpia es la misma red evaluada con y sin máscara.
3. **La máscara de la v3 era fraccional**; el resumen dice binaria. Con huecos aleatorios, la fraccional no quita
   ninguna celda.
4. **El factorial 2×2 está implementado y probado** (E3c, E3cd, E4c, E4cd; E4cf/E4cfd como ablación; E2c). Falta
   entrenarlo: unas 8-10 h de GPU con `tools/cola_d06.sh`.
5. **Todo el test de la v3 es exploratorio.** La mezcla E1u + E2u (49.70) es un resultado complementario de
   ensamble, no de MLCA.

## Cómo seguir

```bash
pip install -r requirements.txt
pytest -q                                        # 91 en CPU; las 2 de GPU, con CUDA
python tools/verificar_defm.py --nyuv2 --n 64    # D-05 con los pesos oficiales
python tools/reproducir_historicos.py            # la v4 reproduce la v3
# escribir en DECISIONES.md (D-06) las predicciones del equipo
bash tools/cola_d06.sh 1                         # factorial, E2c, tablas, coste y diagnósticos
```

Uso de IA: la auditoría, el código nuevo, las pruebas y estos documentos se hicieron con Claude (Anthropic), como
permite el curso. Las predicciones y la lectura de los resultados son del equipo.
