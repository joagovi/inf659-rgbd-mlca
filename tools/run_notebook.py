# -*- coding: utf-8 -*-
"""Ejecuta un notebook completo y guarda una copia con las salidas.

Uso:  python tools/run_notebook.py <ruta/al/notebook.ipynb>

Guarda <nombre>_ejecutado.ipynb junto al original, con salidas y figuras
embebidas (ignorado por git). El notebook limpio es el que se versiona.
A diferencia de run_cells.py, no está atado a un notebook concreto.
"""
import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nb_path = Path(sys.argv[1]).resolve()
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                        resources={"metadata": {"path": str(nb_path.parent)}},
                        allow_errors=True)
print(f"ejecutando {nb_path.name} ...")
client.execute()

out = nb_path.with_name(nb_path.stem + "_ejecutado.ipynb")
nbformat.write(nb, out)
print(f"guardado: {out.name}\n")

errores = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errs:
        errores += 1
        print(f"celda {i:2d}  ERROR  {errs[0].ename}: {str(errs[0].evalue)[:300]}")
    else:
        print(f"celda {i:2d}  ok")
print("\nRESULTADO:", "todas las celdas OK" if not errores else f"{errores} celdas con error")
sys.exit(1 if errores else 0)
