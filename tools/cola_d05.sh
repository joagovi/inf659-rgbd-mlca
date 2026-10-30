#!/usr/bin/env bash
# Cola de D-05 (DECISIONES.md), desde la raíz del repositorio:   bash tools/cola_d05.sh
# 1. E2u: si su mejor lr sigue en el extremo del barrido, prueba la siguiente (y, si cambia, repite sus 3 semillas).
# 2. E10 (rasgos de DeFM que dependen de la entrada en las tres escalas) y su mezcla con E1u.
# 3. E11 (E10 con la profundidad también en el codificador) y su mezcla con E1u.
# 4. Si E2u cambió de lr, repite E1u + E2u (combinar.py guarda aparte el resultado anterior).
# Necesita las corridas de E1u y E2u en runs/rgbd. Unas 5 a 7 horas en una RTX 4060 Laptop.
# Uso de IA: escrito con asistencia de IA (Claude, Anthropic); ver DECISIONES.md.
# D-06: `set -euo pipefail` (antes solo `set -u`: un fallo de E2u no detenía la cola, y un error de Python en la
# comprobación final se confundía con «la lr no cambió»).
set -euo pipefail
export PYTHONIOENCODING=utf-8
date
python src/entrenar_rgbd.py --completo E2u
date
python src/entrenar_rgbd.py --completo E10 && python src/combinar.py --color E1u --fusion E10
date
python src/entrenar_rgbd.py --completo E11 && python src/combinar.py --color E1u --fusion E11
date
CAMBIO=$(python -c "import sys; sys.path.insert(0, 'src'); import entrenar_rgbd as E; print(int(E.elegida('E2u')[0][0] != 3e-3))")
if [ "$CAMBIO" = 1 ]; then
    python src/combinar.py --color E1u --fusion E2u
fi
date
