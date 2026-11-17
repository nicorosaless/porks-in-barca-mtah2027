import numpy as np

from mtah.hibrido.decodificar import D, Costes, decodificar
from mtah.predictores.hibrido import seccion_con_saques


def test_frames_antes_del_saque_salen_D_y_la_posesion_arranca_en_el_saque():
    t = 200
    seccion = np.zeros(t, np.int8)
    seccion[20:180] = 2
    sec = seccion_con_saques(seccion, [[20, 60]])
    assert (sec[20:60] == 0).all() and (sec[60:180] == 2).all()
    assert (seccion[20:60] == 2).all()  # no muta la entrada
    s = (t + 4) // 5
    logp = np.log(np.tile([0.98, 0.01, 0.01], (s, 1)))  # todo H
    out = decodificar(logp, sec, Costes())
    assert (out[20:60] == D).all()
    assert (out[60:180] != D).all()


def test_sin_saques_devuelve_la_misma_seccion():
    seccion = np.array([0, 1, 1, 2, 2], np.int8)
    assert (seccion_con_saques(seccion, None) == seccion).all()
