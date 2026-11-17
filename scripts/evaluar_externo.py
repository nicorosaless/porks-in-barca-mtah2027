#!/usr/bin/env python3
"""Evalúa predictores en los partidos IDSSE, limpios y degradados, con el scorer oficial.

Es la validación del equipo: aquí se eligen métodos y parámetros. El GT público del reto (5 min)
NO se usa para elegir nada (RULES.md, issue #16).

    python scripts/evaluar_externo.py                          # todos los predictores
    python scripts/evaluar_externo.py baseline_smooth50 ...    # solo esos
    python scripts/evaluar_externo.py --semillas 3 --limpio

Recuerda: diferencias de spellF1 por debajo de ~0,04 son ruido (EVALUATION.md).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from mtah.convencion import MINIMO_S_DEFECTO, posesion_controlada  # noqa: E402
from mtah.degradar import Degradacion, degradar  # noqa: E402
from mtah.evaluar import puntuar  # noqa: E402
from mtah.escenarios import ESCENARIOS  # noqa: E402
from mtah.fuentes import idsse, metrica  # noqa: E402
from mtah.predictores import PREDICTORES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictores", nargs="*", default=list(PREDICTORES))
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--semillas", type=int, default=2, help="degradaciones por partido")
    ap.add_argument("--limpio", action="store_true", help="evalúa también sin degradar")
    ap.add_argument("--pesimista", action="store_true",
                    help="balón muerto nunca ausente (ver scripts/hibrido/entrenar.py)")
    ap.add_argument("--fuente", choices=("idsse", "metrica"), default="idsse")
    ap.add_argument("--escenarios", default=None)
    ap.add_argument("--convencion", choices=("dfl", "metrica"), default="dfl")
    ap.add_argument("--minimo_s", type=float, default=MINIMO_S_DEFECTO)
    args = ap.parse_args()

    fuente = idsse if args.fuente == "idsse" else metrica
    partidos = args.partidos or fuente.disponibles()
    if not partidos:
        sys.exit("no hay partidos {} disponibles".format(args.fuente))
    escenarios = [x.strip() for x in args.escenarios.split(",") if x.strip()] if args.escenarios else ["pes" if args.pesimista else "v2"]
    for e in escenarios:
        if e not in ESCENARIOS:
            raise SystemExit("escenario desconocido {} (disponibles: {})".format(e, sorted(ESCENARIOS)))
    filas = []  # (predictor, condicion, partido, spell, macro, n_pred, n_gt)
    for m in partidos:
        p = fuente.cargar(m)
        if args.fuente == "idsse" and args.convencion == "metrica":
            p = replace(p, gt=posesion_controlada(p.gt, p.seccion, int(round(args.minimo_s * p.fps))))
        condiciones = [("limpio", p)] if args.limpio else []
        for esc in escenarios:
            extra = ESCENARIOS[esc]
            if args.fuente == "metrica":
                extra = {**extra, "balon_ausente": 0.0, "quitar_congelado": False}
            condiciones += [("{}_s{}".format(esc, s), degradar(p, Degradacion(semilla=s, **extra)))
                            for s in range(args.semillas)]
        for cond, q in condiciones:
            for nombre in args.predictores:
                t0 = time.time()
                r = puntuar(q.gt, PREDICTORES[nombre](q))
                filas.append((nombre, cond, m, r["spell_f1_mean"], r["macro_f1"],
                              r["n_pred_spells"], r["n_gt_spells"], time.time() - t0))

    print("\n{:<24} {:<7} {:>8} {:>8} {:>9} {:>7}".format(
        "predictor", "cond", "spellF1", "macroF1", "pred/gt", "partidos"))
    for nombre in args.predictores:
        for cond in sorted({f[1] for f in filas}, reverse=True):
            sel = [f for f in filas if f[0] == nombre and f[1] == cond]
            if not sel:
                continue
            spell = np.array([f[3] for f in sel])
            print("{:<24} {:<7} {:>8.4f} {:>8.4f} {:>9.2f} {:>7}  (spellF1 min {:.3f} max {:.3f})".format(
                nombre, cond, spell.mean(), np.mean([f[4] for f in sel]),
                np.sum([f[5] for f in sel]) / np.sum([f[6] for f in sel]),
                len({f[2] for f in sel}), spell.min(), spell.max()))


if __name__ == "__main__":
    main()
