"""Iteración 1 del híbrido: emisiones a mano + máquina de estados (sin aprendizaje).

Por paso, la evidencia de cada estado:

    H, A   cercanía del jugador más cercano de cada equipo al balón de referencia
           (-dist / tau), con el balón interpolado cuando falta
    D      constante + bonus si el balón está fuera del campo + bonus si falta el balón,
           más fuerte cuanto más largo es el hueco

Luego `decodificar` con costes de cambio. Todos los pesos se ajustan en IDSSE.
"""
from dataclasses import dataclass

import numpy as np

from mtah.hibrido import rasgos as R


@dataclass(frozen=True)
class Pesos:
    tau: float = 2.0          # metros
    base_d: float = -3.0
    fuera: float = 6.0
    falta: float = 1.0
    hueco: float = 1.0        # por unidad de log1p(segundos sin balón)/4
    lento: float = 0.0        # bonus de D si el balón casi no se mueve


def emisiones(r, w: Pesos):
    """rasgos -> log-evidencias [S, 3] (H, A, D), sin normalizar."""
    jug = r["jug"].astype(np.float32)
    vis = jug[..., 0] > 0
    local = jug[..., 1] > 0
    dist = jug[..., 8] * 20.0
    inf = np.float32(60.0)
    dh = np.where(vis & local, dist, inf).min(axis=1)
    da = np.where(vis & ~local, dist, inf).min(axis=1)
    b = r["balon"]
    bv, fuera = b[:, 0], b[:, 6]
    hueco = np.maximum(b[:, 7], b[:, 8]) * (1 - bv)
    vel = np.hypot(b[:, 3], b[:, 4]) * 20.0
    lento = (b[:, 5] > 0) & (vel < 0.5)
    ld = (w.base_d + w.fuera * fuera + w.falta * (1 - bv) + w.hueco * hueco * 4.0
          + w.lento * lento)
    return np.stack([-dh / w.tau, -da / w.tau, ld], axis=1).astype(np.float32)
