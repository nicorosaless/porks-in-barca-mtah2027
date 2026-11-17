#!/usr/bin/env python3
"""Compara convenciones de etiquetado (solo agregados): GT público de BAR-ATH, IDSSE y Metrica sample.

CHALLENGE.md: "the question that matters is ... what the corrected tracking means by 'dead' and
by 'owning' — and the public ground truth is where you can see it". Este script describe la
convención (duraciones, transiciones, dónde está el balón cuando empieza/acaba D); no ajusta nada.

    python scripts/convencion_publica.py [--fuentes metrica,skillcorner]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.fuentes import idsse, metrica, reto, skillcorner  # noqa: E402


def tramos(gt):
    """[(ini, fin, etiqueta)] de rachas de gt (0 H, 1 A, 2 D, 3 X)."""
    cambio = np.flatnonzero(np.diff(gt)) + 1
    ini = np.concatenate(([0], cambio))
    fin = np.concatenate((cambio, [len(gt)]))
    return [(a, b, int(gt[a])) for a, b in zip(ini, fin)]


def fuera(b):
    x, y = b
    return x < 0 or x > 1 or y < 0 or y > 1


def ultimo_visto(balon, f, hacia, maximo=50):
    """Posición del balón observada más cercana a f en dirección `hacia` (-1 atrás, +1 delante)."""
    for k in range(maximo):
        g = f + hacia * k
        if 0 <= g < len(balon) and not np.isnan(balon[g, 0]):
            return balon[g], k
    return None, None


def describir(p, gt):
    fps = p.fps
    ts = [t for t in tramos(gt) if t[2] != 3]
    pos = [(b - a) / fps for a, b, e in ts if e in (0, 1)]
    dd = [(b - a) / fps for a, b, e in ts if e == 2]
    vivo = sum(pos) / 60
    # transiciones entre tramos consecutivos (sin X entre medias)
    directas = sum(1 for u, v in zip(ts, ts[1:]) if u[1] == v[0] and {u[2], v[2]} == {0, 1})
    via_d = sum(1 for u, v, w in zip(ts, ts[1:], ts[2:])
                if v[2] == 2 and u[2] in (0, 1) and w[2] in (0, 1))
    via_d_mismo = sum(1 for u, v, w in zip(ts, ts[1:], ts[2:])
                      if v[2] == 2 and u[2] in (0, 1) and w[2] == u[2])
    # balón al empezar D: fuera del campo (última posición vista antes o en el frame)
    ini_fuera, ini_falta, dist_lin = [], [], []
    fin_falta, vel_tras = [], []
    for a, b, e in ts:
        if e != 2:
            continue
        pb, k = ultimo_visto(p.balon, a, -1)
        ini_falta.append(k is None or k > 0)
        if pb is not None:
            ini_fuera.append(fuera(pb))
            x, y = pb
            dist_lin.append(min(abs(x), abs(1 - x)) * p.campo[0] if 0.1 < y < 0.9 or True else 0)
        # al acabar D: balón visible y velocidad en el primer segundo vivo
        fin_falta.append(np.isnan(p.balon[min(b, len(gt) - 1), 0]))
        seg = p.balon[b:b + fps] * np.array(p.campo)
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1) * fps
        d = d[~np.isnan(d)]
        if len(d):
            vel_tras.append(np.max(d))
    r = dict(
        pos=len(pos), pos_min=np.mean(pos) if pos else np.nan, pos_med=np.median(pos) if pos else np.nan,
        pos_corta=np.mean(np.array(pos) < 3) if pos else np.nan,
        pos_min_s=np.min(pos) if pos else np.nan,
        pos_x_min_vivo=len(pos) / vivo if vivo else np.nan,
        fr_d=np.mean(gt[gt != 3] == 2), n_d=len(dd), d_media=np.mean(dd) if dd else np.nan,
        d_med=np.median(dd) if dd else np.nan,
        directas=directas, via_d=via_d, via_d_mismo=via_d_mismo,
        d_ini_fuera=np.mean(ini_fuera) if ini_fuera else np.nan,
        d_ini_falta=np.mean(ini_falta) if ini_falta else np.nan,
        d_fin_falta=np.mean(fin_falta) if fin_falta else np.nan,
        vel_max_tras=np.median(vel_tras) if vel_tras else np.nan,
    )
    return r


def imprimir(nombre, r):
    print("{:<14} pos {:>4}  media {:>5.1f}s  mediana {:>5.1f}s  <3s {:>4.0%}  min {:>4.1f}s  "
          "pos/min vivo {:>4.1f} | D {:>4.0%} n {:>3} media {:>5.1f}s mediana {:>5.1f}s | "
          "H<->A directas {:>3} via D {:>3} (mismo eq. {:>3}) | D empieza: fuera {:>4.0%} "
          "sin balón {:>4.0%} | D acaba: sin balón {:>4.0%}  vmax 1er s {:>5.1f} m/s".format(
              nombre, r["pos"], r["pos_min"], r["pos_med"], r["pos_corta"], r["pos_min_s"],
              r["pos_x_min_vivo"], r["fr_d"], r["n_d"], r["d_media"], r["d_med"], r["directas"],
              r["via_d"], r["via_d_mismo"], r["d_ini_fuera"], r["d_ini_falta"], r["d_fin_falta"],
              r["vel_max_tras"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuentes", default="reto,idsse,metrica,skillcorner",
                    help="subconjunto de reto,idsse,metrica,skillcorner")
    fuentes = set(ap.parse_args().fuentes.replace("+", ",").split(","))
    if "reto" in fuentes:
        p = reto.cargar(gt_publico=True)
        imprimir("BAR-ATH pub", describir(p, p.gt))
    filas = {}
    for nombre, mod, corto in (("idsse", idsse, "IDSSE"), ("metrica", metrica, "Metrica"),
                               ("skillcorner", skillcorner, "SkillC")):
        if nombre not in fuentes:
            continue
        for m in mod.disponibles():
            q = mod.cargar(m)
            r = describir(q, q.gt)
            filas.setdefault(corto, []).append(r)
            imprimir("{} {}".format(corto, m), r)
    for corto, rs in filas.items():
        print("{:<8} n {:>2} | pos/min vivo {:.2f} [{:.2f}-{:.2f}]  <3s {:.0%} [{:.0%}-{:.0%}]  "
              "D {:.0%} [{:.0%}-{:.0%}]  D media {:.1f}s [{:.1f}-{:.1f}]".format(
                  corto, len(rs), *[f(r[k] for r in rs) for k in ("pos_x_min_vivo", "pos_corta",
                                                                  "fr_d", "d_media")
                                    for f in (lambda v: np.mean(list(v)), min, max)]))


if __name__ == "__main__":
    main()
