#!/usr/bin/env python3
"""Mezcla las log P fuera de muestra de varios runs: P = sum_i w_i P_i, para afinar.py.

    python scripts/hibrido/ensamblar.py salidas/hibrido/runs/it8 \
        salidas/hibrido/runs/it7 salidas/hibrido/runs/it4c+salidas/hibrido/runs/it4c_x --pesos 0.5,0.5

Cada run puede ser una lista de carpetas unidas con "+" (se busca el fichero en orden). Los nombres
salen de todas las carpetas del primer run; en los demás se prueba también sin el prefijo "idsse:" (runs antiguos).
"""
import argparse
import glob
import os

import numpy as np


def logsumexp(x, axis, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    r = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return r if keepdims else np.squeeze(r, axis=axis)


def buscar(carpetas, nombre):
    for c in carpetas:
        for n in (nombre, nombre.replace("oof_idsse:", "oof_")):
            ruta = os.path.join(c, n)
            if os.path.exists(ruta):
                return ruta
    return None


def mezclar(logps, pesos):
    pila = np.stack([lp - logsumexp(lp, axis=1, keepdims=True) for lp in logps])
    return logsumexp(pila + np.log(np.asarray(pesos, float))[:, None, None], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pesos", required=True)
    args = ap.parse_args()
    pesos = [float(x) for x in args.pesos.split(",")]
    assert len(pesos) == len(args.runs) and abs(sum(pesos) - 1) < 1e-9
    runs = [r.split("+") for r in args.runs]
    os.makedirs(args.salida, exist_ok=True)
    hechos = faltan = 0
    nombres = sorted({os.path.basename(r) for c in runs[0]
                      for r in glob.glob(os.path.join(c, "oof_*.npz"))})
    for nombre in nombres:
        rutas = [buscar(c, nombre) for c in runs]
        if None in rutas:
            faltan += 1
            continue
        logps = [np.load(r)["logp"] for r in rutas]
        assert len({lp.shape for lp in logps}) == 1, (nombre, [lp.shape for lp in logps])
        np.savez_compressed(os.path.join(args.salida, nombre), logp=mezclar(logps, pesos))
        hechos += 1
    print("ensamblados {}  sin pareja {}".format(hechos, faltan))


if __name__ == "__main__":
    main()
