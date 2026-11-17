#!/usr/bin/env python3
"""Compara agregados del balón: ATD del reto frente a IDSSE degradado. Sin etiquetas del reto.

Solo se miran estadísticas del input (visibilidad, fracción fuera del campo, histograma de
velocidad). Sirve para elegir los parámetros de balón congelado de mtah/degradar.py.

    python scripts/hibrido/calibrar_balon.py [--rejilla]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mtah.degradar import Degradacion, degradar  # noqa: E402
from mtah.fuentes import idsse, reto  # noqa: E402

BORDES = [0, 0.05, 0.5, 1, 2, 5, 10, 20, 1e9]


def agregados(p):
    b = p.balon * np.array(p.campo)
    v = np.linalg.norm(b[2:] - b[:-2], axis=1) * p.fps / 2
    ok = ~np.isnan(v) & (p.seccion[1:-1] > 0)
    x, y = p.balon[1:-1, 0], p.balon[1:-1, 1]
    fuera = ((x < 0) | (x > 1) | (y < 0) | (y > 1))[ok]
    ev = p.seccion > 0
    h = np.histogram(v[ok], BORDES)[0] / ok.sum()
    return np.concatenate([[(~np.isnan(p.balon[ev, 0])).mean(), fuera.mean()], h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejilla", action="store_true")
    ap.add_argument("--partidos", nargs="*", default=["J03WMX", "J03WQQ", "J03WR9"])
    args = ap.parse_args()
    atd = agregados(reto.cargar(gt_publico=False))
    partidos = [idsse.cargar(m) for m in args.partidos]
    etiqueta = "vis fuera | <.05 .5 1 2 5 10 20 >20 m/s"
    print("{:<44} {}".format("", etiqueta))
    print("{:<44} {}".format("ATD", np.round(atd, 3)))

    def probar(cfg):
        a = np.mean([agregados(degradar(p, cfg)) for p in partidos], axis=0)
        err = np.abs(a - atd).sum()
        return a, err

    candidatos = [Degradacion(quitar_congelado=False), Degradacion()]
    if args.rejilla:
        for ca, cf, vm in itertools.product([0.3, 0.5, 0.6, 0.7, 0.8], [0.8, 0.9, 0.95, 1.0],
                                            [1.0, 1.5, 2.5, 4.0]):
            candidatos.append(Degradacion(congelado_ausente=ca, congelado_fuera_ausente=cf,
                                          congelado_vel_mediana=vm))
    res = []
    for cfg in candidatos:
        a, err = probar(cfg)
        res.append((err, cfg, a))
        nombre = "original" if not cfg.quitar_congelado else "aus {} fuera {} vel {}".format(
            cfg.congelado_ausente, cfg.congelado_fuera_ausente, cfg.congelado_vel_mediana)
        print("{:<44} {}  L1 {:.3f}".format(nombre, np.round(a, 3), err), flush=True)
    err, cfg, a = min(res, key=lambda r: r[0])
    print("mejor: aus {} fuera {} vel {} L1 {:.3f}".format(
        cfg.congelado_ausente, cfg.congelado_fuera_ausente, cfg.congelado_vel_mediana, err))


if __name__ == "__main__":
    main()
