import numpy as np

from mtah.convencion import posesion_controlada
from mtah.partido import A, D, H


def _gt(*bloques):
    vals = []
    for etq, n in bloques:
        vals.extend([etq] * n)
    gt = np.array(vals, np.int8)
    sec = np.ones(len(gt), np.int8)
    return gt, sec


def test_absorbe_toque_corto_entre_posesiones_largas():
    gt, sec = _gt((A, 100), (H, 30), (A, 100))
    assert (posesion_controlada(gt, sec, 50) == A).all()


def test_no_absorbe_si_viene_de_balon_muerto():
    gt, sec = _gt((A, 100), (D, 50), (H, 30), (A, 100))
    out = posesion_controlada(gt, sec, 50)
    np.testing.assert_array_equal(out, gt)


def test_no_absorbe_si_es_largo():
    gt, sec = _gt((A, 100), (H, 60), (A, 100))
    np.testing.assert_array_equal(posesion_controlada(gt, sec, 50), gt)


def test_fusiona_segmentos_contiguos_mismo_equipo():
    gt, sec = _gt((A, 100), (H, 30), (H, 30))
    np.testing.assert_array_equal(posesion_controlada(gt, sec, 50), gt)


def test_absorbe_cadena_de_toques_cortos():
    gt, sec = _gt((A, 100), (H, 20), (A, 20), (H, 20), (A, 100))
    assert (posesion_controlada(gt, sec, 50) == A).all()


def test_absorbe_toque_corto_antes_de_salir():
    gt, sec = _gt((A, 100), (H, 30), (D, 50))
    out = posesion_controlada(gt, sec, 50)
    assert (out[:130] == A).all()
    assert (out[130:] == D).all()
