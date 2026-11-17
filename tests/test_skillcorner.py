import numpy as np

from mtah.fuentes.skillcorner import Fase, etiquetas_desde_fases, interpolar_10_a_25, norm_xy
from mtah.partido import A, D, H

_ = -1  # sin equipo conocido en ese frame


def test_d_fuera_de_fases_y_forward_fill_dentro():
    fases = [Fase(2, 6, H), Fase(6, 9, A), Fase(12, 16, H)]
    equipos = np.array([H, H, _, _, A, _, _, H, _, A, A, _, _, _, A, _, _, _], np.int8)
    gt = etiquetas_desde_fases(fases, equipos, 18)
    assert (gt[:2] == D).all()               # antes de la primera fase, aunque haya equipo
    assert (gt[2:4] == A).all()              # back-fill: primer equipo del tramo
    assert (gt[4:7] == A).all()              # forward-fill sigue al cruzar fases contiguas
    assert (gt[7:9] == H).all()
    assert (gt[9:12] == D).all()             # hueco entre fases = D aunque haya eventos
    assert (gt[12:16] == A).all()            # tramo nuevo: back-fill desde su propio evento
    assert (gt[16:] == D).all()


def test_tramo_sin_equipo_usa_respaldo_y_despues_la_fase():
    fases = [Fase(0, 4, A), Fase(6, 10, A)]
    equipos = np.full(10, _, np.int8)
    respaldo = np.array([_, _, _, _, _, _, _, H, _, _], np.int8)
    gt = etiquetas_desde_fases(fases, equipos, 10, respaldo=respaldo)
    assert (gt[0:4] == A).all()              # sin eventos ni grupo: equipo de la fase
    assert (gt[4:6] == D).all()
    assert (gt[6:10] == H).all()             # grupo del tracking, back-fill


def test_posesion_disputada_no_cambia_hasta_el_control_rival():
    fases = [Fase(0, 10, H)]
    equipos = np.array([H, H, _, _, _, _, A, A, _, _], np.int8)
    gt = etiquetas_desde_fases(fases, equipos, 10)
    assert (gt[:6] == H).all() and (gt[6:] == A).all()


def test_interpolar_10_a_25_tiempo_y_valores():
    x = np.arange(5, dtype=np.float32)[:, None]      # 0..4 a 10 fps = 0.4 s
    y = interpolar_10_a_25(x)
    assert y.shape == (11, 1)                         # 0.4 s a 25 fps
    assert np.allclose(y[:, 0], np.arange(11) * 0.4)  # x(t) = 10 t


def test_interpolar_cruza_hueco_de_un_frame_pero_no_de_dos():
    x = np.array([0, 1, np.nan, 3, np.nan, np.nan, 6, 7], np.float32)[:, None]
    y = interpolar_10_a_25(x)[:, 0]
    t10 = np.arange(len(y)) * 0.4                     # posición en frames de 10 fps
    puente = (t10 >= 1) & (t10 <= 3)
    assert np.allclose(y[puente], t10[puente])        # un frame ausente: se interpola
    hueco = (t10 > 3) & (t10 < 6)
    assert np.isnan(y[hueco]).all()                   # dos frames ausentes: NaN
    assert np.allclose(y[t10 >= 6], t10[t10 >= 6])


def test_norm_xy_origen_arriba_izquierda():
    xy = np.array([[-52.5, 34.0], [52.5, -34.0], [0.0, 0.0]], np.float32)
    out = norm_xy(xy, 105.0, 68.0)
    assert np.allclose(out, [[0, 0], [1, 1], [0.5, 0.5]])
