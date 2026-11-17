import itertools

import numpy as np

import mtah.hibrido.decodificar as dec
from mtah.hibrido.decodificar import PASO, Costes, decodificar, viterbi
from mtah.partido import D


def _viterbi_original(logp, c):
    """Copia literal del Viterbi anterior a las duraciones mínimas (referencia)."""
    s = len(logp)
    if s == 0:
        return np.zeros(0, np.int8)
    e = logp * c.temperatura
    e = e + np.array([0.0, 0.0, c.sesgo_d])
    trans = -np.array([[0.0, c.cambio, c.a_muerto],
                       [c.cambio, 0.0, c.a_muerto],
                       [c.desde_muerto, c.desde_muerto, 0.0]])
    atras = np.zeros((s, 3), np.int8)
    v = e[0].copy()
    for t in range(1, s):
        cand = v[:, None] + trans
        o = cand.argmax(axis=0)
        atras[t] = o
        v = cand[o, np.arange(3)] + e[t]
    out = np.zeros(s, np.int8)
    out[-1] = int(v.argmax())
    for t in range(s - 1, 0, -1):
        out[t - 1] = atras[t, out[t]]
    return out


def _rachas(x):
    """[(ini, fin, valor)] de rachas consecutivas de x."""
    cortes = np.flatnonzero(np.diff(x) != 0) + 1
    bordes = [0, *cortes.tolist(), len(x)]
    return [(a, b, int(x[a])) for a, b in zip(bordes[:-1], bordes[1:])]


def _secciones(rng, t):
    sec = np.zeros(t, np.int8)
    a = int(rng.integers(0, 40))
    m = int(rng.integers(t // 3, 2 * t // 3))
    b = int(rng.integers(m + 1, m + 60))
    sec[a:m] = 1
    sec[b:t - int(rng.integers(0, 40))] = 2
    return sec


def _costes(rng, **kw):
    return Costes(cambio=float(rng.uniform(0, 6)), a_muerto=float(rng.uniform(0, 6)),
                  desde_muerto=float(rng.uniform(0, 6)), sesgo_d=float(rng.uniform(-1, 1)),
                  temperatura=float(rng.uniform(0.5, 2)), **kw)


def test_sin_minimos_igual_que_el_original():
    rng = np.random.default_rng(10)
    for _ in range(20):
        t = int(rng.integers(500, 3000))
        sec = _secciones(rng, t)
        logp = rng.normal(size=((t - 1) // PASO + 1, 3))
        c = _costes(rng)
        assert c.min_pos == 0 and c.min_d == 0
        assert viterbi(logp, c).tobytes() == _viterbi_original(logp, c).tobytes()
        orig = dec.viterbi
        dec.viterbi = _viterbi_original
        try:
            esperado = decodificar(logp, sec, c)
        finally:
            dec.viterbi = orig
        assert decodificar(logp, sec, c).tobytes() == esperado.tobytes()


def test_minimo_uno_equivale_a_sin_minimo():
    # min 1 no restringe nada pero pasa por el código expandido
    rng = np.random.default_rng(11)
    for _ in range(20):
        logp = rng.normal(size=(int(rng.integers(1, 400)), 3))
        c = _costes(rng, min_pos=1, min_d=1)
        assert (viterbi(logp, c) == _viterbi_original(logp, c)).all()


def test_con_minimos_no_hay_rachas_interiores_cortas():
    rng = np.random.default_rng(12)
    for _ in range(15):
        t = int(rng.integers(1000, 4000))
        sec = _secciones(rng, t)
        logp = rng.normal(size=((t - 1) // PASO + 1, 3)) * 2
        min_pos, min_d = int(rng.integers(2, 16)), int(rng.integers(2, 26))
        c = _costes(rng, min_pos=min_pos, min_d=min_d)
        c = Costes(0.5, 0.5, 0.5, c.sesgo_d, c.temperatura, min_pos, min_d)  # costes bajos: muchos cambios
        est = viterbi(logp[:600], c)
        rs = _rachas(est)
        assert len(rs) > 2
        for a, b, v in rs[1:-1]:
            assert b - a >= (min_d if v == 2 else min_pos)
        out = decodificar(logp, sec, c)
        cortes = np.flatnonzero(np.diff(sec) != 0) + 1
        for a, b in zip([0, *cortes], [*cortes, t]):
            if sec[a] == 0:
                assert (out[a:b] == D).all()
                continue
            for x, y, v in _rachas(out[a:b])[1:-1]:
                assert y - x >= PASO * (min_d if v == D else min_pos)


def _puntuacion(est, logp, c):
    e = logp * c.temperatura + np.array([0.0, 0.0, c.sesgo_d])
    trans = -np.array([[0.0, c.cambio, c.a_muerto],
                       [c.cambio, 0.0, c.a_muerto],
                       [c.desde_muerto, c.desde_muerto, 0.0]])
    return e[np.arange(len(est)), est].sum() + trans[est[:-1], est[1:]].sum()


def _factible(est, c):
    return all(b - a >= (c.min_d if v == 2 else c.min_pos) for a, b, v in _rachas(est)[1:-1])


def test_coincide_con_fuerza_bruta():
    rng = np.random.default_rng(13)
    for caso in range(40):
        s = int(rng.integers(1, 9))
        logp = rng.normal(size=(s, 3)) * 2
        c = _costes(rng, min_pos=int(rng.integers(0, 5)), min_d=int(rng.integers(0, 5)))
        if caso % 2:
            c = Costes(0.3, 0.3, 0.3, c.sesgo_d, c.temperatura, c.min_pos, c.min_d)
        mejor = max(_puntuacion(np.array(x), logp, c)
                    for x in itertools.product(range(3), repeat=s) if _factible(np.array(x), c))
        est = viterbi(logp, c)
        assert _factible(est, c)
        assert np.isclose(_puntuacion(est, logp, c), mejor, rtol=0, atol=1e-9)
