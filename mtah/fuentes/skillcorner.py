"""SkillCorner Open Data -> `Partido`.

Datos: SkillCorner Open Data, https://github.com/SkillCorner/opendata (licencia MIT). 20 partidos
de la A-League 2024/25: tracking JSONL a 10 fps (`tracking_extrapolated.jsonl`), metadatos
(`match.json`), `dynamic_events.csv` y `phases_of_play.csv`. Se descargan con
`scripts/descargar_skillcorner.py` a `$MTAH_EXTERNAL_DIR/skillcorner/`.

Tracking
--------
- SkillCorner da metros con el origen en el centro del campo, x a lo largo e y a lo ancho, con
  y positiva hacia la banda de arriba de la imagen (`image_corners_projection`: las esquinas
  "top" tienen y > 0). Se pasa a 0-1 con origen arriba a la izquierda, como el ATD:
  `x01 = (x + L/2) / L`, `y01 = (W/2 - y) / W`. `campo = (pitch_length, pitch_width)`.
- `is_detected` falso = posición extrapolada (fuera de cámara): se pone a NaN, tanto el balón
  como los jugadores. Así el balón falta de verdad, como en el ATD.
- 10 -> 25 fps por interpolación lineal. Un frame de 25 fps solo recibe valor si los frames de
  10 fps que lo rodean (anterior y siguiente con dato) distan como mucho 2, es decir, se cruzan
  huecos de 1 frame de 10 fps pero no más largos.

Etiquetas (convención del reto, CHALLENGE.md)
---------------------------------------------
- vivo = el frame cae dentro de alguna fase de `phases_of_play` (README: las fases solo existen
  con el balón en juego). Fuera de las fases, D.
- Dentro de cada tramo vivo (unión de fases contiguas), forward-fill del equipo de los eventos
  `player_possession` de `dynamic_events.csv` (posesión controlada por un jugador); back-fill al
  principio del tramo. Si un tramo no tiene ninguno, `possession.group` del tracking y, si
  tampoco, `team_in_possession_id` de su primera fase.
- Se descartó usar `possession.group` como fuente principal: en los 20 partidos coincide con el
  equipo de `player_possession` solo en el 87 % de los frames y con el jugador más cercano al
  balón en el 73 % (eventos: 78 %). Ver `scripts/skillcorner_agregados.py` y docs/HIBRIDO.md.
"""
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

from mtah.partido import A, D, H, X, Partido
from mtah.rutas import cache_dir, external_dir

SEPARACION = 25
FPS_ORIGEN = 10
FPS_DESTINO = 25


@dataclass(frozen=True)
class Fase:
    inicio: int        # frame 10 fps, incluido
    fin: int           # frame 10 fps, excluido
    equipo: int = -1   # H/A de team_in_possession_id, -1 si no hay


def _dir():
    return os.path.join(external_dir(), "skillcorner")


def _ruta(mid, sufijo):
    return os.path.join(_dir(), "matches", str(mid), "{}_{}".format(mid, sufijo))


def _cache(mid):
    return os.path.join(cache_dir(), "skillcorner_{}.npz".format(mid))


def disponibles():
    """Ids (str) con caché o con los ficheros necesarios, ordenados."""
    ids = set()
    raiz = os.path.join(_dir(), "matches")
    if os.path.isdir(raiz):
        for mid in os.listdir(raiz):
            if mid.isdigit() and all(os.path.exists(_ruta(mid, s)) for s in (
                    "match.json", "tracking_extrapolated.jsonl", "phases_of_play.csv")):
                ids.add(mid)
    if os.path.isdir(cache_dir()):
        for f in os.listdir(cache_dir()):
            if f.startswith("skillcorner_") and f.endswith(".npz"):
                ids.add(f[len("skillcorner_"):-4])
    return sorted(ids, key=int)


def norm_xy(xy, largo, ancho):
    """Metros centrados (y hacia arriba) -> 0-1 con origen arriba a la izquierda."""
    xy = np.asarray(xy, np.float32)
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] + largo / 2) / largo
    out[..., 1] = (ancho / 2 - xy[..., 1]) / ancho
    return out


def interpolar_10_a_25(x, n_destino=None):
    """Interpola [n, ...] de 10 a 25 fps; NaN donde el hueco original pasa de 1 frame.

    El frame j de destino está en t = j / 25 s, es decir, en la posición s = j * 0.4 del origen.
    Se toma el último dato válido en floor(s) o antes y el primero en ceil(s) o después; si están a
    más de 2 frames (hueco de 2 o más frames ausentes), queda NaN.
    """
    x = np.asarray(x, np.float32)
    n = len(x)
    if n_destino is None:
        n_destino = (n - 1) * FPS_DESTINO // FPS_ORIGEN + 1 if n else 0
    plano = x.reshape(n, -1)
    out = np.full((n_destino, plano.shape[1]), np.nan, np.float32)
    if n == 0 or n_destino == 0:
        return out.reshape((n_destino,) + x.shape[1:])
    j = np.arange(n_destino)
    num = j * FPS_ORIGEN                       # s = num / FPS_DESTINO, aritmética entera exacta
    lo = np.minimum(num // FPS_DESTINO, n - 1)
    hi = np.minimum(lo + (num % FPS_DESTINO > 0), n - 1)
    s = num / FPS_DESTINO
    idx = np.arange(n)
    for c in range(plano.shape[1]):
        v = plano[:, c]
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        prev = np.maximum.accumulate(np.where(ok, idx, -1))
        nxt = np.minimum.accumulate(np.where(ok, idx, n)[::-1])[::-1]
        a = prev[lo]
        b = nxt[hi]
        valido = (a >= 0) & (b < n) & (b - a <= 2)
        a_, b_ = a[valido], b[valido]
        va, vb = v[a_], v[b_]
        den = np.where(b_ > a_, b_ - a_, 1)
        w = np.where(b_ > a_, (s[valido] - a_) / den, 0.0)
        out[valido, c] = va + (vb - va) * w
    return out.reshape((n_destino,) + x.shape[1:])


def _tramos(mascara):
    """[(ini, fin)] de rachas True."""
    m = np.concatenate(([False], np.asarray(mascara, bool), [False]))
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def _rellenar(g):
    """Forward-fill de los valores >= 0 y back-fill antes del primero. None si no hay ninguno."""
    ok = np.flatnonzero(g >= 0)
    if len(ok) == 0:
        return None
    pos = np.maximum.accumulate(np.where(g >= 0, np.arange(len(g)), -1))
    pos[pos < 0] = ok[0]
    return g[pos]


def etiquetas_desde_fases(fases, equipos, n_frames, respaldo=None):
    """H/A/D por frame de 10 fps.

    `fases`: lista de `Fase` (fin excluido). `equipos`: int8[n] con H, A o -1 por frame (el equipo
    que controla el balón cuando se sabe). Fuera de fases, D. Dentro de cada tramo vivo (unión de
    fases contiguas), forward-fill de `equipos` y back-fill al principio. Si el tramo no tiene
    ningún valor, se usa `respaldo` (mismo formato) y después el equipo de su primera fase; si
    tampoco hay, el tramo queda D.
    """
    equipos = np.asarray(equipos, np.int8)
    vivo = np.zeros(n_frames, bool)
    equipo_fase = np.full(n_frames, -1, np.int8)
    for f in fases:
        a, b = max(0, int(f.inicio)), min(n_frames, int(f.fin))
        if b > a:
            vivo[a:b] = True
            sel = equipo_fase[a:b]
            sel[sel < 0] = f.equipo
    gt = np.full(n_frames, D, np.int8)
    for a, b in _tramos(vivo):
        for fuente in (equipos, respaldo, equipo_fase):
            if fuente is None:
                continue
            r = _rellenar(np.asarray(fuente[a:b], np.int8))
            if r is not None:
                gt[a:b] = r
                break
    return gt


def equipo_eventos(mid, meta, n_frames):
    """int8[n]: equipo (H/A) de cada frame dentro de un `player_possession`, -1 fuera."""
    team = {int(meta["home_team"]["id"]): H, int(meta["away_team"]["id"]): A}
    filas = []
    with open(_ruta(mid, "dynamic_events.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") == "player_possession" and row.get("team_id"):
                e = team.get(int(float(row["team_id"])))
                if e is not None:
                    filas.append((int(row["frame_start"]), int(row["frame_end"]) + 1, e))
    out = np.full(n_frames, -1, np.int8)
    for a, b, e in sorted(filas):
        out[max(0, a):min(n_frames, b)] = e
    return out


def _codigo_grupo(g):
    if g in ("home", "home team"):
        return H
    if g in ("away", "away team"):
        return A
    return -1


def leer_meta(mid):
    with open(_ruta(mid, "match.json")) as f:
        return json.load(f)


def leer_fases(mid, meta):
    home, away = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    fases = []
    with open(_ruta(mid, "phases_of_play.csv"), newline="") as f:
        for row in csv.DictReader(f):
            t = row.get("team_in_possession_id") or ""
            t = int(float(t)) if t else None
            fases.append(Fase(int(row["frame_start"]), int(row["frame_end"]) + 1,
                              H if t == home else A if t == away else -1))
    return fases


def _slots(meta):
    home_id = int(meta["home_team"]["id"])
    jugadores = sorted(meta["players"], key=lambda p: int(p["id"]))
    ids = [int(p["id"]) for p in jugadores]
    equipo = np.array([0 if int(p["team_id"]) == home_id else 1 for p in jugadores], np.int8)
    return ids, {pid: i for i, pid in enumerate(ids)}, equipo


def leer_tracking(mid, meta):
    """Arrays a 10 fps indexados por `frame`: balón [n,2], jugadores [n,K,2] (metros, NaN si no
    detectado), periodo int8[n] (0 si no hay), grupo int8[n] (H/A/-1)."""
    _, slot, _ = _slots(meta)
    filas = []
    with open(_ruta(mid, "tracking_extrapolated.jsonl")) as f:
        for line in f:
            if line.strip():
                filas.append(json.loads(line))
    n = max(int(r["frame"]) for r in filas) + 1
    balon = np.full((n, 2), np.nan, np.float32)
    jug = np.full((n, len(slot), 2), np.nan, np.float32)
    periodo = np.zeros(n, np.int8)
    grupo = np.full(n, -1, np.int8)
    for r in filas:
        i = int(r["frame"])
        if r.get("period") in (1, 2):
            periodo[i] = r["period"]
        grupo[i] = _codigo_grupo((r.get("possession") or {}).get("group"))
        b = r.get("ball_data") or {}
        if b.get("is_detected") is True and b.get("x") is not None:
            balon[i] = (b["x"], b["y"])
        for pj in r.get("player_data") or []:
            if pj.get("is_detected") is True and pj.get("x") is not None:
                k = slot.get(int(pj["player_id"]))
                if k is not None:
                    jug[i, k] = (pj["x"], pj["y"])
    return balon, jug, periodo, grupo


def convertir(mid):
    meta = leer_meta(mid)
    largo, ancho = float(meta["pitch_length"]), float(meta["pitch_width"])
    _, _, equipo_slot = _slots(meta)
    balon10, jug10, periodo10, grupo10 = leer_tracking(mid, meta)
    n10 = len(balon10)
    gt10 = etiquetas_desde_fases(leer_fases(mid, meta), equipo_eventos(mid, meta, n10), n10,
                                 respaldo=grupo10)
    out_b, out_j, out_sec, out_gt = [], [], [], []
    for per in (1, 2):
        idx = np.flatnonzero(periodo10 == per)
        if len(idx) == 0:
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        b25 = interpolar_10_a_25(norm_xy(balon10[a:b], largo, ancho))
        j25 = interpolar_10_a_25(norm_xy(jug10[a:b], largo, ancho))
        n25 = len(b25)
        # etiqueta del frame de 10 fps más cercano en el tiempo
        src = np.minimum((np.arange(n25) * 2 * FPS_ORIGEN + FPS_DESTINO) // (2 * FPS_DESTINO), b - a - 1)
        out_b += [b25, np.full((SEPARACION, 2), np.nan, np.float32)]
        out_j += [j25, np.full((SEPARACION, j25.shape[1], 2), np.nan, np.float32)]
        out_sec += [np.full(n25, per, np.int8), np.zeros(SEPARACION, np.int8)]
        out_gt += [gt10[a:b][src], np.full(SEPARACION, X, np.int8)]
    p = Partido("skillcorner_{}".format(mid), (largo, ancho), np.concatenate(out_b),
                np.concatenate(out_j), equipo_slot, np.concatenate(out_sec),
                np.concatenate(out_gt).astype(np.int8), fps=FPS_DESTINO)
    return p.validar()


def cargar(mid, usar_cache=True):
    mid = str(mid)
    ruta = _cache(mid)
    if usar_cache and os.path.exists(ruta):
        return Partido.cargar(ruta)
    os.makedirs(cache_dir(), exist_ok=True)
    p = convertir(mid)
    p.guardar(ruta)
    return p
