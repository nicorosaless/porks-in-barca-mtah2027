"""Rasgos por paso (5 fps) a partir de un `Partido`. Los mismos para IDSSE y para el reto.

Solo numpy: se calculan igual en MareNostrum (datos externos) y en local (reto).

Balón de referencia: el observado si lo hay; si no, interpolación lineal entre la última y la
siguiente vez que se vio (se pueden usar frames futuros, issue #15). Los jugadores se describen
respecto a ese balón y se quedan los KV visibles más cercanos a él, ordenados por distancia.

    balon  [S, FB]      vis, x, y, vx, vy, vel_ok, fuera, t_desde, t_hasta, lx, ly, nx, ny
    jug    [S, KV, FJ]  vis, local, x, y, vx, vy, dx, dy, dist
    etiq   [S]          H/A/D, o -1 si no se puntúa (X o fuera de sección)
    balon_real [S, 2]   balón de referencia sin degradar (objetivo auxiliar), NaN si no hay

Coordenadas en metros centradas y escaladas: x / 52.5, y / 34.
"""
import numpy as np

from mtah.hibrido.decodificar import PASO
from mtah.partido import X

KV = 26
FB = 13
FJ = 9
DT = 5  # frames a cada lado para la velocidad


def _metros(p, xy):
    return (xy - 0.5) * np.array(p.campo, np.float32)


def _escala(m):
    return m / np.array([52.5, 34.0], np.float32)


def balon_referencia(p):
    """[T, 2] en metros, más índices del último y siguiente frame con balón."""
    b = _metros(p, p.balon.astype(np.float32))
    t = len(b)
    vis = ~np.isnan(b[:, 0])
    idx = np.arange(t)
    ult = np.maximum.accumulate(np.where(vis, idx, -1))
    sig = np.minimum.accumulate(np.where(vis, idx, t)[::-1])[::-1]
    ref = b.copy()
    falta = ~vis
    tiene_u, tiene_s = ult >= 0, sig < t
    u = np.clip(ult, 0, t - 1)
    s = np.clip(sig, 0, t - 1)
    ambos = falta & tiene_u & tiene_s
    w = np.where(ambos, (idx - u) / np.maximum(s - u, 1), 0.0)[:, None]
    ref[ambos] = ((1 - w) * b[u] + w * b[s])[ambos]
    solo_u = falta & tiene_u & ~tiene_s
    ref[solo_u] = b[u][solo_u]
    solo_s = falta & ~tiene_u & tiene_s
    ref[solo_s] = b[s][solo_s]
    ref[np.isnan(ref)] = 0.0
    return b, ref, vis, ult, sig


def rasgos(p, gt=None, limpio=None):
    """Devuelve dict con balon, jug, etiq, seccion_paso, balon_real (si `limpio`)."""
    t = p.n_frames
    s_n = (t - 1) // PASO + 1
    f = np.minimum(np.arange(s_n) * PASO + PASO // 2, t - 1)
    fa, fb = np.maximum(f - DT, 0), np.minimum(f + DT, t - 1)
    dt_s = (fb - fa) / p.fps

    b, ref, vis, ult, sig = balon_referencia(p)
    bv = vis[f]
    vel_ok = vis[fa] & vis[fb] & (dt_s > 0)
    vb = np.where(vel_ok[:, None], (b[fb] - b[fa]) / np.maximum(dt_s, 1e-3)[:, None], 0.0)
    vb = np.nan_to_num(vb) / 20.0
    crudo = p.balon[f]
    fuera = bv & ((crudo[:, 0] < 0) | (crudo[:, 0] > 1) | (crudo[:, 1] < 0) | (crudo[:, 1] > 1))
    t_desde = np.where(ult[f] >= 0, (f - ult[f]) / p.fps, 60.0)
    t_hasta = np.where(sig[f] < t, (sig[f] - f) / p.fps, 60.0)
    u = np.clip(ult[f], 0, t - 1)
    s = np.clip(sig[f], 0, t - 1)
    lpos = np.where((ult[f] >= 0)[:, None], np.nan_to_num(b[u]), 0.0)
    npos = np.where((sig[f] < t)[:, None], np.nan_to_num(b[s]), 0.0)
    rf = ref[f]
    balon = np.concatenate([
        bv[:, None], _escala(rf), vb, vel_ok[:, None], fuera[:, None],
        np.log1p(np.minimum(t_desde, 60.0))[:, None] / 4.0,
        np.log1p(np.minimum(t_hasta, 60.0))[:, None] / 4.0,
        _escala(lpos), _escala(npos)], axis=1).astype(np.float32)

    jm = _metros(p, p.jugadores[f].astype(np.float32))           # [S, K, 2]
    ja = _metros(p, p.jugadores[fa].astype(np.float32))
    jb = _metros(p, p.jugadores[fb].astype(np.float32))
    jvis = ~np.isnan(jm[..., 0])
    vok = ~np.isnan(ja[..., 0]) & ~np.isnan(jb[..., 0])
    vj = np.where(vok[..., None], (jb - ja) / np.maximum(dt_s, 1e-3)[:, None, None], 0.0)
    vj = np.nan_to_num(vj) / 10.0
    rel = jm - rf[:, None, :]
    dist = np.sqrt((rel ** 2).sum(-1))
    dist = np.where(jvis, dist, np.inf)
    orden = np.argsort(dist, axis=1, kind="stable")[:, :KV]       # [S, KV]
    fila = np.arange(s_n)[:, None]
    k = min(KV, jm.shape[1])
    orden = orden[:, :k]
    v = jvis[fila, orden]
    local = np.where(p.equipo_slot[orden] == 0, 1.0, -1.0)
    jug = np.zeros((s_n, KV, FJ), np.float32)
    jug[:, :k, 0] = v
    jug[:, :k, 1] = local * v
    jug[:, :k, 2:4] = np.nan_to_num(_escala(jm[fila, orden])) * v[..., None]
    jug[:, :k, 4:6] = vj[fila, orden] * v[..., None]
    jug[:, :k, 6:8] = np.nan_to_num(_escala(rel[fila, orden])) * v[..., None]
    jug[:, :k, 8] = np.where(v, np.minimum(dist[fila, orden], 60.0) / 20.0, 0.0)

    sec = p.seccion[f]
    out = {"balon": balon, "jug": jug.astype(np.float16), "seccion": sec.astype(np.int8)}
    g = p.gt if gt is None else gt
    if g is not None:
        e = g[f].astype(np.int8)
        e = np.where((e == X) | (sec == 0), -1, e).astype(np.int8)
        out["etiq"] = e
    if limpio is not None:
        out["balon_real"] = _escala(_metros(limpio, limpio.balon[f].astype(np.float32)))
    return out
