"""Estadísticas agregadas de calidad del tracking, para comparar el ATD del reto con datos
externos degradados. Solo devuelve números agregados (fracciones, cuantiles): nada fila a fila.
"""
import numpy as np

CUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def _rachas(mascara):
    """Longitudes de las rachas de True en un vector booleano."""
    m = np.concatenate(([False], mascara, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def calidad(partido, solo_evaluable=True):
    sel = partido.seccion > 0 if solo_evaluable else np.ones(partido.n_frames, bool)
    falta = np.isnan(partido.balon[:, 0])
    huecos = np.concatenate([_rachas(falta & (partido.seccion == s)) for s in (1, 2)])
    vis = ~np.isnan(partido.jugadores[:, :, 0])
    frag = np.concatenate([_rachas(vis[:, k] & sel) for k in range(vis.shape[1])])
    return {
        "frames": int(sel.sum()),
        "slots": int(vis.shape[1]),
        "balon_ausente": float(falta[sel].mean()),
        "huecos_balon_n": int(len(huecos)),
        "huecos_balon_q": dict(zip(CUANTILES, np.quantile(huecos, CUANTILES).round(1).tolist()))
        if len(huecos) else {},
        "jugadores_visibles_media": float(vis[sel].sum(1).mean()),
        "fragmentos_n": int(len(frag)),
        "fragmentos_q": dict(zip(CUANTILES, np.quantile(frag, CUANTILES).round(1).tolist()))
        if len(frag) else {},
    }


def imprimir(nombre, c):
    print("{}: frames={} slots={} balón ausente={:.1%} jugadores visibles={:.1f}".format(
        nombre, c["frames"], c["slots"], c["balon_ausente"], c["jugadores_visibles_media"]))
    print("   huecos de balón (frames) n={} cuantiles={}".format(c["huecos_balon_n"], c["huecos_balon_q"]))
    print("   fragmentos de pista (frames) n={} cuantiles={}".format(c["fragmentos_n"], c["fragmentos_q"]))
