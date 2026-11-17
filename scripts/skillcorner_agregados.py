#!/usr/bin/env python3
"""Verifica la derivación de etiquetas de SkillCorner con agregados (sin filas crudas).

Compara, en los 20 partidos y a 10 fps, dos fuentes del equipo en posesión dentro de los tramos
vivos (fases de juego):
  grupo    forward-fill de `possession.group` del tracking (propuesta inicial)
  eventos  forward-fill del equipo de los `player_possession` de dynamic_events.csv (elegida)

Para cada una imprime:
  - concordancia con `player_possession` (frames de posesión de jugador con la misma etiqueta);
  - concordancia con una señal independiente: equipo del jugador detectado más cercano al balón
    detectado, cuando está a < 1.5 m;
  - posesiones/min de balón vivo, % de posesiones < 3 s, % D y duración media de D;
  - si los tramos D corresponden a interrupciones: fracción de tramos D precedidos por una
    posesión con `game_interruption_after` o seguidos de una con `game_interruption_before`.

    python scripts/skillcorner_agregados.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.fuentes import skillcorner as S  # noqa: E402
from mtah.partido import A, D, H  # noqa: E402

FPS = S.FPS_ORIGEN


def _posesiones(mid, meta):
    team = {int(meta["home_team"]["id"]): H, int(meta["away_team"]["id"]): A}
    out = []
    with open(S._ruta(mid, "dynamic_events.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") == "player_possession" and row.get("team_id"):
                out.append((int(row["frame_start"]), int(row["frame_end"]) + 1,
                            team[int(float(row["team_id"]))],
                            bool(row.get("game_interruption_before")),
                            bool(row.get("game_interruption_after"))))
    return out


def _rachas(gt, periodo):
    out = []
    for per in (1, 2):
        idx = np.flatnonzero(periodo == per)
        g = gt[idx[0]:idx[-1] + 1]
        c = np.flatnonzero(np.diff(g)) + 1
        out += [(int(g[a]), (b - a) / FPS, idx[0] + a, idx[0] + b)
                for a, b in zip(np.r_[0, c], np.r_[c, len(g)])]
    return out


def describir(gt, periodo, pos, cercano):
    ev = np.full(len(gt), -1, np.int8)
    for a, b, e, _, _ in pos:
        ev[a:b] = e
    m = ev >= 0
    mc = cercano >= 0
    r = _rachas(gt, periodo)
    p = np.array([d for e, d, _, _ in r if e in (H, A)])
    dd = [(d, a, b) for e, d, a, b in r if e == D]
    ev_tras = {a for a, _, _, ib, _ in pos if ib}
    ev_ant = {b for _, b, _, _, ia in pos if ia}
    marcado = 0
    for _, a, b in dd:
        if any(abs(x - a) <= 2 * FPS for x in ev_ant) or any(abs(x - b) <= 2 * FPS for x in ev_tras):
            marcado += 1
    vivo_min = p.sum() / 60
    evaluable = periodo > 0
    return dict(conc_ev=(gt[m] == ev[m]).mean(), conc_cerca=(gt[mc] == cercano[mc]).mean(),
                pos_min=len(p) / vivo_min, corta=(p < 3).mean(), fr_d=(gt[evaluable] == D).mean(),
                d_media=np.mean([d for d, _, _ in dd]), d_hora=len(dd) / (evaluable.sum() / FPS / 3600),
                d_interrupcion=marcado / len(dd))


def main():
    filas = {"grupo": [], "eventos": []}
    for mid in S.disponibles():
        meta = S.leer_meta(mid)
        balon, jug, periodo, grupo = S.leer_tracking(mid, meta)
        fases = S.leer_fases(mid, meta)
        pos = _posesiones(mid, meta)
        _, _, eq = S._slots(meta)
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(jug - balon[:, None, :], axis=2)
            dh = np.nanmin(np.where(eq[None] == 0, d, np.inf), 1)
            da = np.nanmin(np.where(eq[None] == 1, d, np.inf), 1)
        cercano = np.where(np.minimum(dh, da) < 1.5, np.where(dh < da, H, A), -1).astype(np.int8)
        n = len(grupo)
        variantes = {
            "grupo": S.etiquetas_desde_fases(fases, grupo, n),
            "eventos": S.etiquetas_desde_fases(fases, S.equipo_eventos(mid, meta, n), n, respaldo=grupo),
        }
        for nombre, gt in variantes.items():
            r = describir(gt, periodo, pos, cercano)
            filas[nombre].append(r)
            print("{} {:<8} concordancia player_possession {:.1%}  jugador cercano {:.1%}  "
                  "pos/min vivo {:.2f}  <3s {:.0%}  D {:.0%}  D media {:.1f}s  D/h {:.0f}  "
                  "D en interrupción {:.0%}".format(
                      mid, nombre, r["conc_ev"], r["conc_cerca"], r["pos_min"], r["corta"],
                      r["fr_d"], r["d_media"], r["d_hora"], r["d_interrupcion"]), flush=True)
    for nombre, rs in filas.items():
        def col(k):
            v = np.array([r[k] for r in rs])
            return v.mean(), v.min(), v.max()
        print("MEDIA {:<8} ".format(nombre) + "  ".join(
            "{} {:.3f} [{:.3f}-{:.3f}]".format(k, *col(k)) for k in rs[0]))


if __name__ == "__main__":
    main()
