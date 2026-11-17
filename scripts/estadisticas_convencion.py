#!/usr/bin/env python3
"""Agregados para elegir T de posesión controlada sin mirar BAR-ATH."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.convencion import posesion_controlada  # noqa: E402
from mtah.degradar import tramos_congelados  # noqa: E402
from mtah.fuentes import idsse, metrica  # noqa: E402
from mtah.partido import A, D, H  # noqa: E402


def _tramos(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def agregados(p, gt=None):
    g = p.gt if gt is None else gt
    ev = p.seccion > 0
    vivo = ev & ((g == H) | (g == A))
    muerto = ev & (g == D)
    minutos_vivos = vivo.sum() / p.fps / 60
    poses = _tramos(vivo & np.r_[True, g[1:] == g[:-1]])
    poses = []
    i = 0
    while i < len(g):
        if vivo[i]:
            j = i + 1
            while j < len(g) and vivo[j] and g[j] == g[i]:
                j += 1
            poses.append((i, j))
            i = j
        else:
            i += 1
    muertos = _tramos(muerto)
    dur_pos = np.array([(b - a) / p.fps for a, b in poses], float)
    dur_d = np.array([(b - a) / p.fps for a, b in muertos], float)
    horas = ev.sum() / p.fps / 3600
    return {
        "frac_d": float(muerto.mean()),
        "d_hora": float(len(muertos) / max(horas, 1e-9)),
        "dur_d": float(dur_d.mean()) if len(dur_d) else 0.0,
        "pos_parte": float(len(poses) / max(len(set(p.seccion[ev])), 1)),
        "pos_min_vivo": float(len(poses) / max(minutos_vivos, 1e-9)),
        "dur_pos": float(dur_pos.mean()) if len(dur_pos) else 0.0,
        "pos_menor_3": float((dur_pos < 3).mean()) if len(dur_pos) else 0.0,
    }


def _media(lista, campo):
    return float(np.mean([x[campo] for x in lista]))


def _print_tabla(nombre, filas):
    print("\n{}".format(nombre))
    print("partido fracD D/h durD pos/parte pos/min vivo durPos pos<3s")
    for k, a in filas:
        print("{:<10} {:.3f} {:.1f} {:.1f} {:.1f} {:.2f} {:.1f} {:.3f}".format(
            k, a["frac_d"], a["d_hora"], a["dur_d"], a["pos_parte"], a["pos_min_vivo"],
            a["dur_pos"], a["pos_menor_3"]))
    print("{:<10} {:.3f} {:.1f} {:.1f} {:.1f} {:.2f} {:.1f} {:.3f}".format(
        "media", *[_media([a for _, a in filas], c)
                   for c in ("frac_d", "d_hora", "dur_d", "pos_parte",
                             "pos_min_vivo", "dur_pos", "pos_menor_3")]))


def _metrica_extra(p):
    ev = p.seccion > 0
    nan = np.isnan(p.balon[ev, 0]).mean()
    congelado = np.zeros(p.n_frames, bool)
    for a, b in tramos_congelados(p.balon):
        congelado[a:b] = True
    frac = congelado[ev].mean()
    coincide = ((p.gt == D) & congelado & ev).sum() / max((congelado & ev).sum(), 1)
    return nan, frac, coincide


def main():
    ids = [idsse.cargar(m) for m in idsse.disponibles()]
    if not ids:
        raise SystemExit("no hay IDSSE")
    originales = [(p.nombre, agregados(p)) for p in ids]
    _print_tabla("IDSSE original", originales)
    medias_t = {}
    for t in (1, 2, 3):
        filas = []
        cambios = []
        for p in ids:
            gt = posesion_controlada(p.gt, p.seccion, int(round(t * p.fps)))
            filas.append((p.nombre, agregados(p, gt)))
            ev = p.seccion > 0
            cambios.append(float((gt[ev] != p.gt[ev]).mean()))
        _print_tabla("IDSSE GT-M T={}s (frames cambiados {:.3%})".format(t, np.mean(cambios)), filas)
        medias_t[t] = _media([a for _, a in filas], "pos_min_vivo")

    mets = []
    for m in metrica.disponibles():
        p = metrica.cargar(m)
        mets.append((m, agregados(p)))
        nan, cong, coincide = _metrica_extra(p)
        print("Metrica {}: balón NaN {:.3f}, congelado {:.3f}, congelado∩D {:.3f}".format(
            m, nan, cong, coincide))
    if mets:
        _print_tabla("Metrica sample", mets)
        objetivo = _media([a for _, a in mets], "pos_min_vivo")
        elegido = min((1, 2, 3), key=lambda t: abs(medias_t[t] - objetivo))
        print("\nT elegido: {}s (pos/min vivo Metrica {:.2f})".format(elegido, objetivo))
    else:
        print("\nMetrica no disponible: fallback T elegido: 2s")


if __name__ == "__main__":
    main()
