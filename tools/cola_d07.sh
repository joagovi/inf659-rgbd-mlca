#!/usr/bin/env bash
# Cola de D-07 (DECISIONES.md), desde la raíz del repositorio:
#     bash tools/cola_d07.sh            # todo, en orden
#     bash tools/cola_d07.sh 2          # desde el paso 2
# Como la de D-06: se detiene en el primer error, una corrida cortada se reanuda al relanzar (estado.pt cada
# 500 pasos) y nada se repite (las corridas terminadas se reconocen por su huella). Se puede apagar la máquina en
# cualquier momento: se pierde a lo sumo lo entrenado desde el último estado.pt.
# Tiempos aproximados en una RTX 4060 Laptop (8 GB).
# Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver DECISIONES.md.
set -euo pipefail
export PYTHONIOENCODING=utf-8
DESDE=${1:-1}
paso() { [ "$1" -ge "$DESDE" ]; }
log() { echo "== $(date '+%F %T') · $*"; }

if paso 1; then   # E11 robusto en un solo modelo (unas 1.7-2 h)
  log "1. E11d (barrido + 3 semillas)"
  python src/entrenar_rgbd.py --completo E11d
fi
if paso 2; then   # la suma con el dropout de E4cd (unas 1.6-1.8 h)
  log "2. E2cd (barrido + 3 semillas)"
  python src/entrenar_rgbd.py --completo E2cd
fi
log "fin"
