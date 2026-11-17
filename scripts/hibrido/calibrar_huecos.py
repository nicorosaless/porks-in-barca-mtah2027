#!/usr/bin/env python3
"""Calibra dónde empiezan los huecos de balón usando solo agregados del input."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mtah.degradar import Degradacion, degradar  # noqa: E402
from mtah.fuentes import idsse, reto  # noqa: E402
from scripts.hibrido.calibrar_balon import agregados as agregados_balon  # noqa: E402


def _tramos_falta(p):
    falta = np.isnan(p.balon[:, 0]) & (p.seccion > 0)
    d = np.diff(np.concatenate(([0], falta.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _dmin_y_cerca(p, indices):
    bm = p.balon[indices] * np.array(p.campo, np.float32)
    jm = p.jugadores[indices] * np.array(p.campo, np.float32)
    d = np.sqrt(((jm - bm[:, None, :]) ** 2).sum(-1))
    return np.nanmin(d, axis=1), np.sum(d < 5.0, axis=1)


def agregados_huecos(p):
    tr = _tramos_falta(p)
    if not tr:
        return np.zeros(7, float)
    ini = np.array([max(a - 1, 0) for a, _ in tr])
    x, y = p.balon[ini, 0], p.balon[ini, 1]
    area = ((x < 0.17) | (x > 0.83)) & (y > 0.2) & (y < 0.8)
    fuera = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    obs = ~np.isnan(x)
    dmin, cerca5 = _dmin_y_cerca(p, ini)
    dmin = np.where(np.isfinite(dmin), dmin, np.nan)
    largos = np.array([b - a for a, b in tr])
    frames_falta = sum(b - a for a, b in tr)
    frames_largos = sum((b - a) for a, b in tr if (b - a) > 20 * p.fps)
    return np.array([
        float(area[obs].mean()) if obs.any() else 0.0,
        float(fuera[obs].mean()) if obs.any() else 0.0,
        float(np.nanmedian(dmin) / 5.0) if np.isfinite(dmin).any() else 0.0,
        float(np.nanmean(cerca5) / 5.0) if len(cerca5) else 0.0,
        float((largos < p.fps).mean()),
        float(((largos >= p.fps) & (largos < 5 * p.fps)).mean()),
        float(frames_largos / max(frames_falta, 1)),
    ])


def vector(p):
    return np.concatenate([agregados_huecos(p), agregados_balon(p)])


def main():
    objetivo = vector(reto.cargar(gt_publico=False))
    print("ATD agregados huecos:", np.round(objetivo[:7], 3))
    print("objetivo referencia plan:", np.round([0.30, 0.15, 1.45 / 5, 1.60 / 5, 0.11, 0.007, 0.67], 3))
    partidos = [idsse.cargar(m) for m in ["J03WMX", "J03WQQ", "J03WR9"]]
    mejores = []
    for pc, pa, pf, cfa in itertools.product([0, 2, 4, 8], [0, 1, 2, 4], [0, 2, 4],
                                             [0.5, 0.7, 0.9]):
        cfg = Degradacion(hueco_peso_cerca=pc, hueco_peso_area=pa, hueco_peso_fuera=pf,
                          congelado_fuera_ausente=cfa, hueco_corto_prob=0.1,
                          hueco_minimo_largo=125)
        a = np.mean([vector(degradar(p, cfg)) for p in partidos], axis=0)
        err = float(np.abs(a - objetivo).sum())
        mejores.append((err, cfg, a))
    err, cfg, a = min(mejores, key=lambda x: x[0])
    print("mejor L1 {:.3f}: {}".format(err, cfg))
    print("agregados mejor:", np.round(a[:7], 3))


if __name__ == "__main__":
    main()
