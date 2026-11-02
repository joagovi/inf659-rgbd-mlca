#!/usr/bin/env bash
# Cola de D-08 (DECISIONES.md), desde la raíz del repositorio:
#     bash tools/cola_d08.sh            # todo, en orden
#     bash tools/cola_d08.sh 2          # desde el paso 2
# Como las de D-06 y D-07: se detiene en el primer error, una corrida cortada se reanuda al relanzar (estado.pt cada
# 500 pasos) y nada se repite (las corridas terminadas se reconocen por su huella). Se puede apagar la máquina en
# cualquier momento: se pierde a lo sumo lo entrenado desde el último estado.pt.
# Tiempos aproximados en una RTX 4060 Laptop (8 GB).
# Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver DECISIONES.md.
set -euo pipefail
export PYTHONIOENCODING=utf-8
DESDE=${1:-1}
paso() { [ "$1" -ge "$DESDE" ]; }
log() { echo "== $(date '+%F %T') · $*"; }

if paso 1; then   # la MLCA entrenada con los huecos reales del sensor (unas 1.8 h)
  log "1. E4er (barrido + 3 semillas)"
  python src/entrenar_rgbd.py --completo E4er
fi
if paso 2; then   # la suma, entrenada igual (unas 1.8 h)
  log "2. E11r (barrido + 3 semillas)"
  python src/entrenar_rgbd.py --completo E11r
fi
if paso 3; then   # la atención sin máscara, entrenada igual: con huecos al entrenar ya no se deriva (unas 1.8 h)
  log "3. E3er (barrido + 3 semillas)"
  python src/entrenar_rgbd.py --completo E3er
fi
if paso 4; then   # tablas y criterio (sin GPU, minutos)
  log "4. tablas y criterio"
  python src/estadistica.py
  python tools/criterio_d06.py --contrastes E4er-E11r E4er-E3er E11r-E11d E4er-E4ed \
    --condiciones completa cruda aleatoria_50 regiones sin | tee runs/rgbd/criterio_d08.txt
fi
log "fin"
