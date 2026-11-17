"""Tests de scripts/hibrido/diagnostico.py (E0). Ejecutar con PYTHONPATH al worktree T-2611-base."""
import importlib.util
import os

import numpy as np

from mtah.hibrido.decodificar import Costes, decodificar
from mtah.partido import A, D, H
from mtah.rapido import spell_f1

_RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "hibrido", "diagnostico.py")
_spec = importlib.util.spec_from_file_location("diagnostico", _RUTA)
DG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DG)


def _codigos(tramos, n):
    out = np.full(n, D, np.int8)
    for a, b, k in tramos:
        out[a:b] = k
    return out


N = 840
GT = _codigos([(0, 100, H), (120, 220, A), (240, 340, H), (360, 460, A),
               (480, 500, H), (500, 520, A), (520, 700, H), (720, 820, H)], N)
PRED = _codigos([(0, 100, H),            # g0 ok
                 (180, 220, A),          # g1 IoU 0.4 -> borde
                 (240, 340, A),          # g2 equipo
                                         # g3 todo D -> perdida
                 (480, 700, H),          # g4 fundida, g5 equipo, g6 ok (IoU 0.82)
                 (720, 745, H), (750, 775, H)],  # g7 partida
                N)


def test_clasifica_cada_tipo():
    filas = DG.clasificar(GT, PRED)
    assert [f["clase"] for f in filas] == [
        "ok", "borde", "equipo", "perdida", "fundida", "equipo", "ok", "partida"]
    assert filas[1]["iou"] == 0.4


def test_oraculo_ambos_reproduce_gt():
    logp = np.zeros((N // 5, 3), np.float32)
    sec = np.ones(N, np.int8)
    c = Costes(cambio=1, a_muerto=2, desde_muerto=2, temperatura=1.5)
    assert spell_f1(GT, decodificar(DG.logp_oraculo(logp, GT, "ambos"), sec, c)) == 1.0
    # sin oráculo y emisiones planas no hay posesiones
    assert spell_f1(GT, decodificar(logp, sec, c)) == 0.0


def test_oraculo_vivo_no_toca_equipo_y_equipo_no_toca_d():
    rng = np.random.default_rng(0)
    logp = rng.normal(size=(N // 5, 3))
    v = DG.logp_oraculo(logp, GT, "vivo")
    e = DG.logp_oraculo(logp, GT, "equipo")
    muerto = GT[::5] == D
    assert (v[muerto, :2] < -1e5).all() and (v[~muerto, 2] < -1e5).all()
    assert (v[muerto, 2] == logp[muerto, 2]).all()
    assert (e[:, 2] == logp[:, 2]).all()
    assert (e[GT[::5] == H, 1] < -1e5).all() and (e[GT[::5] == A, 0] < -1e5).all()
