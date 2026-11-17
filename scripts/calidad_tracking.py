#!/usr/bin/env python3
"""Compara la calidad del tracking: ATD del reto frente a IDSSE limpio y degradado.

Sirve para ajustar mtah/degradar.py: el IDSSE degradado debería parecerse al ATD en balón
ausente, longitud de huecos, slots y fragmentos. Solo imprime agregados.

    python scripts/calidad_tracking.py [J03WMX]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import estadisticas as est  # noqa: E402
from mtah.degradar import degradar  # noqa: E402
from mtah.fuentes import idsse, reto  # noqa: E402


def main(argv):
    match_id = argv[0] if argv else "J03WMX"
    try:
        est.imprimir("ATD reto", est.calidad(reto.cargar(gt_publico=False)))
    except SystemExit as e:  # games.require sale con un mensaje si no hay datos del reto
        print("ATD reto: no disponible ({})".format(str(e).splitlines()[0]))
    p = idsse.cargar(match_id)
    est.imprimir("IDSSE {} limpio".format(match_id), est.calidad(p))
    est.imprimir("IDSSE {} degradado".format(match_id), est.calidad(degradar(p)))


if __name__ == "__main__":
    main(sys.argv[1:])
