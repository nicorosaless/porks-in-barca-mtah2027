"""Predictores híbridos para el reto: emisiones (heurística o modelo) + Viterbi.

Los parámetros vienen de `modelos/hibrido/<nombre>.json`, que escribe scripts/hibrido/afinar.py
con el ajuste sobre los 7 partidos IDSSE (nunca con el GT público). Si hay `pesos_red`, se carga
la red entrenada en MareNostrum solo con IDSSE y se ejecuta aquí, en local.
"""
import json
import os

import numpy as np

from mtah.hibrido import heuristica as HE
from mtah.hibrido import rasgos as R
from mtah.hibrido.decodificar import Costes, decodificar
from mtah.rutas import RAIZ

MODELOS = os.path.join(RAIZ, "modelos", "hibrido")


def _logp_red(r, pesos_red, convencion, tta=False):
    import torch

    from mtah.hibrido.modelo import Red
    from mtah.hibrido.modelo import predecir as predecir_red
    from mtah.hibrido.modelo import predecir_tta
    ck = torch.load(os.path.join(MODELOS, pesos_red), map_location="cpu")
    red = Red(d=ck["args"]["d"], dt=ck["args"]["dt"],
              convenciones=ck["args"].get("convenciones", 0))
    red.load_state_dict(ck["estado"])
    return (predecir_tta if tta else predecir_red)(red, r, "cpu", 1 if convencion == "metrica" else 0)


def mezclar(logps, pesos):
    """log(sum_i w_i P_i), frame a frame (mezcla de probabilidades, no de log P)."""
    pila = np.stack(logps) + np.log(np.asarray(pesos, float))[:, None, None]
    m = pila.max(0)
    return m + np.log(np.exp(pila - m).sum(0))


def seccion_con_saques(seccion, tramos):
    """Copia de `seccion` con 0 (no evaluable -> D) en cada tramo [ini, fin) anterior a un saque.

    Regla dura de la documentación pública (FAQ.md): en BAR-ATH la segunda parte se evalúa desde el
    frame 110.808, pero no se saca hasta el 113.742, y esos frames son todos D. Al cortar la sección,
    Viterbi arranca de cero en el saque.
    """
    sec = seccion.copy()
    for ini, fin in tramos or []:
        sec[ini:fin] = 0
    return sec


def predictor(nombre):
    def predecir(p):
        cfg = json.load(open(os.path.join(MODELOS, nombre + ".json")))
        r = R.rasgos(p, gt=None)
        if cfg.get("componentes"):
            comp = cfg["componentes"]
            logp = mezclar([_logp_red(r, c["pesos_red"], c.get("convencion"), cfg.get("tta", False))
                            for c in comp],
                           [c["peso"] for c in comp])
        elif cfg.get("pesos_red"):
            logp = _logp_red(r, cfg["pesos_red"], cfg.get("convencion"))
        else:
            logp = HE.emisiones(r, HE.Pesos(**cfg["pesos"]))
        seccion = seccion_con_saques(p.seccion, cfg.get("d_antes_saque"))
        return decodificar(logp, seccion, Costes(**cfg["costes"]))
    predecir.__name__ = "hibrido_" + nombre
    return predecir


def disponibles():
    if not os.path.isdir(MODELOS):
        return []
    return sorted(f[:-5] for f in os.listdir(MODELOS) if f.endswith(".json"))
