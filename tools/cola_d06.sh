#!/usr/bin/env bash
# Cola de D-06 (DECISIONES.md y auditoria/06_plan_experimentos.md), desde la raíz del repositorio:
#     bash tools/cola_d06.sh            # todo, en orden de prioridad
#     bash tools/cola_d06.sh 2          # desde el paso 2
# Se detiene en el primer error (set -e) y una corrida cortada se reanuda al relanzar. Nada se repite: las corridas
# terminadas se reconocen por su huella, y una carpeta con otra configuración detiene la cola (ConfigConflict).
# Tiempos aproximados en una RTX 4060 Laptop (8 GB), unos 17 min por corrida.
# Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver DECISIONES.md.
set -euo pipefail
export PYTHONIOENCODING=utf-8
DESDE=${1:-0}
paso() { [ "$1" -ge "$DESDE" ]; }
log() { echo "== $(date '+%F %T') · $*"; }

if paso 0; then   # comprobaciones, sin entrenar (minutos)
  log "0. pruebas, verificación de DeFM, reproducción de la v3, huecos efectivos"
  python -m pytest -q -m "not lenta"
  python tools/verificar_defm.py --nyuv2 --n 64
  python tools/reproducir_historicos.py
  python tools/huecos_efectivos.py --nyuv2
fi
if paso 1; then   # factorial sin dropout: una sola trayectoria para E4c, E3c y E4cf (unas 1.7-2.5 h)
  log "1. E4c (barrido + 3 semillas) y, sin entrenar, E3c y E4cf"
  python src/entrenar_rgbd.py --completo E4c
  python src/entrenar_rgbd.py --derivar E4c E3c
  python src/entrenar_rgbd.py --derivar E4c E4cf
fi
if paso 2; then   # factorial con dropout (unas 3.5-5 h)
  log "2. E4cd y E3cd (barrido + 3 semillas cada uno) y, sin entrenar, E4cfd"
  python src/entrenar_rgbd.py --completo E4cd
  python src/entrenar_rgbd.py --completo E3cd
  python src/entrenar_rgbd.py --derivar E4cd E4cfd
fi
if paso 3; then   # control de que la derivación vale en la GPU (17 min)
  log "3. E3c entrenado de verdad, semilla 0 (debe dar lo mismo que el derivado, salvo el no determinismo de la GPU)"
  python - <<'PY'
import sys; sys.path.insert(0, "src")
import entrenar_rgbd as E
(lr, _), _ = E.elegida("E4c")
E.entrenar("E3c", lr, 0, nombre=f"control_E3c_entrenado_lr{lr:g}_s0")
PY
fi
if paso 4; then   # E2 con los rasgos corregidos (unas 1.7 h)
  log "4. E2c"
  python src/entrenar_rgbd.py --completo E2c
fi
if paso 5; then   # tablas y diagnósticos (sin GPU para las tablas)
  log "5. tablas, coste y diagnósticos en la validación"
  python src/estadistica.py
  python src/tablas.py --validacion
  python src/coste.py
  python src/diagnosticos_d06.py --exps E4 E5 E4c E3c E4cd E3cd --datos validacion
fi
log "fin"
