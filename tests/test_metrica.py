import numpy as np

from mtah.fuentes.metrica import Evento, _leer_tracking_csv, etiquetas_desde_eventos
from mtah.partido import A, D, H


def test_etiquetas_desde_eventos_convencion_metrica():
    eventos = [
        Evento(0, "SET PIECE", "", 1, 10, 10),
        Evento(0, "PASS", "", 1, 20, 30),
        Evento(0, "BALL LOST", "", 1, 35, 35),
        Evento(1, "RECOVERY", "", 1, 40, 40),
        Evento(1, "BALL OUT", "", 1, 50, 55),
        Evento(0, "SET PIECE", "", 1, 70, 70),
        Evento(1, "FAULT RECEIVED", "", 1, 80, 80),
        Evento(1, "SET PIECE", "", 1, 90, 90),
        Evento(1, "SHOT", "GOAL", 1, 100, 105),
        Evento(0, "SET PIECE", "", 1, 120, 120),
    ]
    gt = etiquetas_desde_eventos(eventos, 140, np.ones(140, np.int8))
    assert (gt[:10] == D).all()
    assert (gt[10:40] == H).all()
    assert gt[40] == A
    assert (gt[55:70] == D).all()
    assert (gt[80:90] == D).all()
    assert (gt[90:105] == A).all()
    assert (gt[105:120] == D).all()
    assert (gt[120:] == H).all()


def test_etiquetas_desde_eventos_paradas_sg3():
    eventos = [
        Evento(0, "SET PIECE", "", 1, 10, 10),
        Evento(0, "SHOT", "OFF TARGET OUT", 1, 20, 25),
        Evento(1, "SET PIECE", "", 1, 40, 40),
        Evento(1, "CHALLENGE", "GROUND FAULT LOST", 1, 50, 50),
        Evento(0, "SET PIECE", "", 1, 70, 70),
        Evento(0, "BALL LOST", "OFFSIDE", 1, 80, 80),
        Evento(1, "SET PIECE", "", 1, 90, 90),
    ]
    gt = etiquetas_desde_eventos(eventos, 110, np.ones(110, np.int8), reglas_sg3=True)
    assert (gt[25:40] == D).all()
    assert (gt[50:70] == D).all()
    assert (gt[80:90] == D).all()
    assert gt[90] == A


def test_leer_tracking_csv_metrica_cabecera_publica(tmp_path):
    cab = [
        ",,,Home,,Home,,,",
        ",,,11,,1,,,",
        "Period,Frame,Time [s],Player11,,Player1,,Ball,",
        "1,1,0.04,0.1,0.2,0.3,0.4,0.5,0.6",
        "1,2,0.08,0.2,0.2,NaN,NaN,0.6,0.6",
        "2,1,0.04,0.7,0.8,0.9,0.1,0.4,0.3",
    ]
    home = tmp_path / "home.csv"
    away = tmp_path / "away.csv"
    home.write_text("\n".join(cab) + "\n")
    away.write_text("\n".join(line.replace("Home", "Away") for line in cab) + "\n")

    bloques, equipo = _leer_tracking_csv(home, away)

    assert equipo.tolist() == [0, 0, 1, 1]
    assert bloques[0][2].shape == (2, 2)
    assert bloques[0][3].shape == (2, 4, 2)
    assert np.isfinite(bloques[0][3][0, :, 0]).sum() == 4
    assert np.isfinite(bloques[0][3][1, :, 0]).sum() == 2
