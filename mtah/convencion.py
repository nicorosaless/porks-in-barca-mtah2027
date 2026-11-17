"""Adaptación de GT de toque (DFL) a posesión controlada."""
import numpy as np

from mtah.partido import A, H

MINIMO_S_DEFECTO = 2.0


def _segmentos(gt, ini, fin):
    i = ini
    while i < fin:
        j = i + 1
        while j < fin and gt[j] == gt[i]:
            j += 1
        yield i, j, int(gt[i])
        i = j


def posesion_controlada(gt, seccion, minimo_frames):
    """Absorbe en la posesión anterior las posesiones H/A cortas que le siguen.

    Regla: una posesión H/A de menos de minimo_frames, precedida sin D ni X por una posesión viva
    del otro equipo, se absorbe en esa posesión anterior. El recorrido es una sola pasada por
    sección; X y cambios de sección rompen la cadena.
    """
    out = np.asarray(gt).copy()
    sec = np.asarray(seccion)
    for s in (1, 2):
        idx = np.flatnonzero(sec == s)
        if len(idx) == 0:
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        previo = None  # (ini, fin, etiqueta) del último segmento fusionado
        for i, j, etq in _segmentos(out, a, b):
            vivo = etq in (H, A)
            if vivo and previo is not None:
                pi, pj, pe = previo
                if pe == etq and pj == i:
                    previo = (pi, j, pe)
                    continue
                if pe in (H, A) and pe != etq and pj == i and (j - i) < minimo_frames:
                    out[i:j] = pe
                    previo = (pi, j, pe)
                    continue
            previo = (i, j, etq) if vivo else None
    return out
