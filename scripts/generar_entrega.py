#!/usr/bin/env python3
"""Genera el CSV de entrega del reto con un predictor y lo valida con el validador oficial.

    python scripts/generar_entrega.py baseline_smooth50 [--salida salidas/equipo.csv]

Imprime el resultado sobre el GT público (5 min) SOLO como comprobación de alineación: si sale
muy por debajo de lo esperado, algo está desplazado. No se usa para elegir entre versiones.
"""
import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.evaluar import puntuar  # noqa: E402
from mtah.fuentes import reto  # noqa: E402
from mtah.partido import escribir_csv  # noqa: E402
from mtah.predictores import PREDICTORES  # noqa: E402
from mtah.rutas import RAIZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictor", choices=sorted(PREDICTORES))
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    salida = args.salida or os.path.join(RAIZ, "salidas", "{}.csv".format(args.predictor))
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    p = reto.cargar(gt_publico=True)
    pred = PREDICTORES[args.predictor](p)
    escribir_csv(salida, pred)

    subprocess.run([sys.executable, os.path.join(RAIZ, "evaluation", "validate_submission.py"),
                    salida], check=True)
    with open(salida, "rb") as f:
        print("sha256 {}  {}".format(hashlib.sha256(f.read()).hexdigest(), salida))
    r = puntuar(p.gt, pred)
    print("comprobación de alineación (GT público, no para elegir): spellF1 {:.4f} macroF1 {:.4f}".format(
        r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()
