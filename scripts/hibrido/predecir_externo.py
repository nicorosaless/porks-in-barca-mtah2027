#!/usr/bin/env python3
"""Escribe log P de un modelo ya entrenado (final.pt / it*.pt) sobre partidos externos que no vio.

Sirve para medir en Metrica sample un modelo entrenado solo con IDSSE (p. ej. it4c) con los mismos
ficheros oof_ que deja entrenar.py, de modo que afinar.py --fuente_datos metrica los puntúe igual.

    python scripts/hibrido/predecir_externo.py modelos/hibrido/it4c.pt salidas/hibrido/runs/it4c \
        --fuente metrica --escenarios v2
"""
import argparse
import os
import sys
from multiprocessing import get_context

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts", "hibrido"))
from entrenar import SEMILLAS_EVAL, _convencion_partido, _folds_k, _preparar  # noqa: E402
from mtah.escenarios import ESCENARIOS, SUFIJO  # noqa: E402
from mtah.fuentes import idsse, metrica, skillcorner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos", help="final.pt / it*.pt, o la carpeta de un run --kfold (usa el fold "
                                  "que no vio cada partido)")
    ap.add_argument("salida")
    ap.add_argument("--fuente", choices=("metrica", "idsse", "skillcorner"), default="metrica")
    ap.add_argument("--escenarios", default="v2")
    ap.add_argument("--convencion_red", type=int, default=None,
                    help="bandera a usar si la red la tiene (por defecto la de la fuente)")
    ap.add_argument("--procesos", type=int, default=4, help="procesos para preparar rasgos")
    ap.add_argument("--tta", action="store_true", help="media de las 8 transformaciones (E8)")
    args = ap.parse_args()

    import torch

    from mtah.hibrido.modelo import Red, predecir, predecir_tta
    prediccion = predecir_tta if args.tta else predecir

    mod = {"idsse": idsse, "metrica": metrica, "skillcorner": skillcorner}[args.fuente]
    redes = {}

    def cargar(ruta):
        if ruta not in redes:
            ck = torch.load(ruta, map_location="cpu")
            a = ck["args"]
            red = Red(d=a["d"], dt=a["dt"], convenciones=a.get("convenciones", 0))
            red.load_state_dict(ck["estado"])
            red.eval()
            redes[ruta] = red
        return redes[ruta]

    if os.path.isdir(args.pesos):
        import json
        a = json.load(open(os.path.join(args.pesos, "args.json")))
        if not a.get("kfold"):
            # leave-one-match-out (it4c): fold_<fuente:partido>.pt es la red que no vio el partido
            def red_de(clave_m):
                ruta = os.path.join(args.pesos, "fold_{}.pt".format(clave_m))
                if not os.path.exists(ruta):  # runs antiguos: fold_<partido>.pt, sin la fuente
                    ruta = os.path.join(args.pesos, "fold_{}.pt".format(clave_m.split(":", 1)[1]))
                return cargar(ruta)
        else:
            fuentes = [x for x in a["fuentes"].replace("+", ",").split(",") if x]
            mods = {"idsse": idsse, "metrica": metrica, "skillcorner": skillcorner}
            todos = ["{}:{}".format(f, m) for f in fuentes for m in mods[f].disponibles()]
            fold_de = {m: k for k, g in _folds_k(todos, a["kfold"]) for m in g}

            def red_de(clave_m):
                return cargar(os.path.join(args.pesos, "fold_{}.pt".format(fold_de[clave_m])))
    else:
        def red_de(clave_m):
            return cargar(args.pesos)
    os.makedirs(args.salida, exist_ok=True)
    escenarios = [e.strip() for e in args.escenarios.replace("+", ",").split(",") if e.strip()]
    tareas = [("{}:{}".format(args.fuente, m), "{}{}".format(s, SUFIJO[esc]), s, ESCENARIOS[esc],
               "dfl", 2.0)
              for m in mod.disponibles() for esc in escenarios for s in SEMILLAS_EVAL]
    torch.set_num_threads(max(1, args.procesos))
    with get_context("spawn").Pool(args.procesos) as pool:
        for clave_m, clave, r in pool.imap(_preparar, tareas):
            conv = _convencion_partido(clave_m) if args.convencion_red is None else args.convencion_red
            logp = prediccion(red_de(clave_m), r, "cpu", conv)
            ruta = os.path.join(args.salida, "oof_{}_s{}.npz".format(clave_m, clave))
            np.savez_compressed(ruta, logp=logp)
            ok = r["etiq"] >= 0
            print("{} s{} acierto por paso {:.4f}".format(
                clave_m, clave, float((logp.argmax(1)[ok] == r["etiq"][ok]).mean())), flush=True)


if __name__ == "__main__":
    main()
