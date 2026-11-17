"""Port del baseline oficial (starter/baseline.py) a la representación común.

Jugador más cercano al balón y voto mayoritario centrado. Sin balón -> D. Debe dar el mismo
fichero que evaluation/results/baseline_smooth*.csv sobre el ATD del reto; tests/ lo comprueba.
Sirve de referencia para que cualquier método nuevo se compare en las mismas condiciones.
"""
import numpy as np

from mtah.partido import A, D, H

NADIE = -1


def mas_cercano(p):
    """Equipo (0/1) del slot más cercano al balón en cada frame; NADIE sin balón o sin jugadores."""
    # float64 y mismo orden de operaciones que el oficial, para que los empates se resuelvan igual
    escala = np.array(p.campo, np.float64)
    dif = p.jugadores.astype(np.float64) * escala - p.balon[:, None, :].astype(np.float64) * escala
    dist = np.einsum("tkc,tkc->tk", dif, dif)
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = np.argmin(dist, axis=1)
    hay = np.isfinite(dist[np.arange(len(k)), k])
    return np.where(hay, p.equipo_slot[k], NADIE).astype(np.int8)


def suavizar(equipos, ventana):
    """Voto mayoritario en ±ventana frames. Empates al local, como el oficial."""
    if ventana <= 0:
        return np.where(equipos == 0, H, np.where(equipos == 1, A, D)).astype(np.int8)
    ch = np.concatenate(([0], np.cumsum(equipos == 0)))
    ca = np.concatenate(([0], np.cumsum(equipos == 1)))
    i = np.arange(len(equipos))
    lo, hi = np.maximum(0, i - ventana), np.minimum(len(equipos), i + ventana + 1)
    h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
    out = np.full(len(equipos), D, np.int8)
    visto = (h + a) > 0
    out[visto] = np.where(h[visto] >= a[visto], H, A)
    return out


def predictor(ventana):
    def predecir(p):
        return suavizar(mas_cercano(p), ventana)
    predecir.__name__ = "baseline_smooth{}".format(ventana) if ventana else "baseline_raw"
    return predecir
