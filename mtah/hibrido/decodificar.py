"""Columna vertebral del híbrido: máquina de estados H/A/D decodificada con Viterbi.

Cada paso (PASO frames) llega con log-probabilidades de H, A y D (de la heurística o del modelo).
El decodificador busca la secuencia de estados más probable con un coste por cambiar de estado:

    H <-> A     cambio directo de posesión          coste `cambio`
    H/A -> D    el balón se para                     coste `a_muerto`
    D -> H/A    se reanuda                           coste `desde_muerto`

Los costes hacen el papel de la histéresis: un cambio solo compensa si la evidencia lo sostiene
durante varios pasos. Es lo que ataca la fragmentación que castiga spellF1. `sesgo_d` mueve el
umbral de balón muerto y `temperatura` escala las emisiones. Todo se ajusta en IDSSE.

`min_pos` y `min_d` fijan la duración mínima (en pasos) de una racha H/A o D: el modelo se expande
a estados (estado, pasos llevados hasta el mínimo) y solo se sale de una racha completa. Las rachas
que tocan el borde de un tramo pueden quedarse cortas. Con ambos a 0 se usa el Viterbi de siempre.
"""
from dataclasses import dataclass

import numpy as np

from mtah.partido import D, H

PASO = 5  # frames por paso: 5 fps


@dataclass(frozen=True)
class Costes:
    cambio: float = 8.0
    a_muerto: float = 6.0
    desde_muerto: float = 6.0
    sesgo_d: float = 0.0
    temperatura: float = 1.0
    min_pos: int = 0  # pasos mínimos de una racha H o A
    min_d: int = 0  # pasos mínimos de una racha D


def viterbi(logp, c: Costes):
    """logp [S, 3] (H, A, D) -> estados [S]. Un tramo por llamada (no cruza la media parte)."""
    s = len(logp)
    if s == 0:
        return np.zeros(0, np.int8)
    if c.min_pos or c.min_d:
        return _viterbi_minimos(logp, c)
    e = logp * c.temperatura
    e = e + np.array([0.0, 0.0, c.sesgo_d])
    trans = -np.array([[0.0, c.cambio, c.a_muerto],
                       [c.cambio, 0.0, c.a_muerto],
                       [c.desde_muerto, c.desde_muerto, 0.0]])  # trans[origen, destino]
    atras = np.zeros((s, 3), np.int8)
    v = e[0].copy()
    for t in range(1, s):
        cand = v[:, None] + trans          # [origen, destino]
        o = cand.argmax(axis=0)
        atras[t] = o
        v = cand[o, np.arange(3)] + e[t]
    out = np.zeros(s, np.int8)
    out[-1] = int(v.argmax())
    for t in range(s - 1, 0, -1):
        out[t - 1] = atras[t, out[t]]
    return out


def _viterbi_minimos(logp, c: Costes):
    """Viterbi con duración mínima por racha, en O(S) sea cual sea el mínimo.

    Por estado j con mínimo m solo hace falta guardar dos cosas por paso:
      ent[t, j]  mejor camino que entra en j justo en t (racha de 1 paso);
      lleno[j]   mejor camino en j con la racha ya completa (>= m pasos o pegada al inicio).
    Los estados intermedios (j, k < m) son ent[t-k+1, j] más las emisiones acumuladas de j,
    así que no se arrastran. Solo se sale de j desde `lleno`.
    """
    s = len(logp)
    e = logp * c.temperatura
    e = e + np.array([0.0, 0.0, c.sesgo_d])
    tr = (-np.array([[0.0, c.cambio, c.a_muerto],
                     [c.cambio, 0.0, c.a_muerto],
                     [c.desde_muerto, c.desde_muerto, 0.0]])).tolist()
    mp, md = max(1, int(round(c.min_pos))), max(1, int(round(c.min_d)))
    mins = (mp, mp, md)
    acum = np.vstack([np.zeros((1, 3)), np.cumsum(e, axis=0)])  # acum[t] = e[:t].sum(0)
    E, A = e.tolist(), acum.tolist()
    inf = -np.inf
    ent = [[inf, inf, inf] for _ in range(s)]
    origen = [[0, 0, 0] for _ in range(s)]        # de qué estado lleno se entró
    cadena = [[False] * 3 for _ in range(s)]      # lleno[t, j] llega por la cadena, no quedándose
    lleno = list(E[0])                            # la racha inicial cuenta como completa
    otros = ((1, 2), (0, 2), (0, 1))
    for t in range(1, s):
        nuevo = [0.0, 0.0, 0.0]
        et = E[t]
        for j in range(3):
            i0, i1 = otros[j]
            v0, v1 = lleno[i0] + tr[i0][j], lleno[i1] + tr[i1][j]
            if v1 > v0:
                v0, i0 = v1, i1
            en = v0 + et[j]
            ent[t][j] = en
            origen[t][j] = i0
            m = mins[j]
            if m == 1:
                quedo, llega = lleno[j] + et[j], en
            else:
                k = t - m + 1
                quedo = lleno[j] + et[j]
                llega = ent[k][j] + A[t + 1][j] - A[k + 1][j] if k >= 1 else inf
            if llega > quedo:
                nuevo[j], cadena[t][j] = llega, True
            else:
                nuevo[j] = quedo
        lleno = nuevo
    # final: racha completa o una última más corta que el mínimo (toca el borde)
    mejor, j, k = max((lleno[j], j, -1) for j in range(3))
    for j2 in range(3):
        for k2 in range(max(1, s - mins[j2] + 1), s):
            v = ent[k2][j2] + A[s][j2] - A[k2 + 1][j2]
            if v > mejor:
                mejor, j, k = v, j2, k2
    out = np.zeros(s, np.int8)
    t = s - 1
    if k >= 0:  # racha final incompleta que entró en k
        out[k:] = j
        t, j = k - 1, origen[k][j]
    while t >= 0:
        if t == 0 or not cadena[t][j]:
            out[t] = j
            t -= 1
            continue
        k = t - mins[j] + 1
        out[k:t + 1] = j
        t, j = k - 1, origen[k][j]
    return out


def tramos(seccion):
    """[(ini, fin)] en frames de cada tramo evaluable continuo (cada parte)."""
    ev = (seccion > 0).astype(np.int8)
    d = np.diff(np.concatenate(([0], ev, [0])))
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for a, b in zip(ini, fin):
        # una parte no se funde con la siguiente aunque no haya hueco entre ellas
        cortes = np.flatnonzero(np.diff(seccion[a:b]) != 0) + 1 + a
        bordes = [a, *cortes.tolist(), b]
        out += list(zip(bordes[:-1], bordes[1:]))
    return out


def decodificar(logp_pasos, seccion, c: Costes):
    """logp por paso [S, 3] con S = ceil(T / PASO) -> códigos por frame [T].

    Los frames no evaluables salen D: el GT tiene X ahí y el scorer no los mira.
    """
    t = len(seccion)
    out = np.full(t, D, np.int8)
    for a, b in tramos(seccion):
        pa, pb = a // PASO, (b - 1) // PASO + 1
        est = viterbi(logp_pasos[pa:pb], c)
        frames = np.arange(a, b) // PASO - pa
        out[a:b] = np.where(est == 0, H, np.where(est == 1, 1, D))[frames]
    return out
