"""Tests de la base. Los que necesitan datos se saltan si no están descargados.

    pytest -q tests
"""
import os

import numpy as np
import pytest

from mtah.degradar import Degradacion, degradar
from mtah.evaluar import puntuar
from mtah.partido import A, D, H, X, Partido
from mtah.predictores import baseline
from mtah.rutas import RAIZ


def partido_sintetico(t=6000, semilla=1):
    rng = np.random.default_rng(semilla)
    jug = rng.uniform(0, 1, (t, 22, 2)).astype(np.float32)
    balon = rng.uniform(0, 1, (t, 2)).astype(np.float32)
    seccion = np.ones(t, np.int8)
    seccion[t // 2 - 25:t // 2] = 0
    seccion[t // 2:] = 2
    gt = np.repeat(np.array([H, A, D, H, A, D], np.int8), t // 6)
    gt[seccion == 0] = X
    return Partido("sint", (105.0, 68.0), balon, jug, np.array([0] * 11 + [1] * 11, np.int8),
                   seccion, gt).validar()


def test_puntuar_perfecto_y_constante():
    p = partido_sintetico()
    assert puntuar(p.gt, p.gt)["spell_f1_mean"] == pytest.approx(1.0)
    # Todo D no tiene posesiones: spellF1 0. (Una constante H no siempre da 0 en un partido
    # sintético con posesiones largas: puede solapar >0.3 con alguna.)
    assert puntuar(p.gt, np.full(p.n_frames, D, np.int8))["spell_f1_mean"] == 0.0


def test_degradar_es_determinista_y_respeta_equipos():
    p = partido_sintetico()
    cfg = Degradacion(balon_ausente=0.4, hueco_balon_mediana=100, fragmento_mediana=300)
    a, b = degradar(p, cfg), degradar(p, cfg)
    np.testing.assert_array_equal(a.balon, b.balon)
    np.testing.assert_array_equal(a.jugadores, b.jugadores)
    ausente = np.isnan(a.balon[p.seccion > 0, 0]).mean()
    assert 0.4 <= ausente < 0.55
    assert a.jugadores.shape[1] > 22                      # más slots que jugadores
    assert (np.bincount(a.equipo_slot) >= 11).all()
    # Cada frame sigue teniendo como mucho 11 jugadores por equipo
    vis = ~np.isnan(a.jugadores[:, :, 0])
    for eq in (0, 1):
        assert vis[:, a.equipo_slot == eq].sum(1).max() <= 11
    assert np.array_equal(a.gt, p.gt)                     # el GT no se toca


def _inicios_huecos(p):
    falta = np.isnan(p.balon[:, 0]) & (p.seccion > 0)
    d = np.diff(np.concatenate(([0], falta.astype(np.int8), [0])))
    return np.flatnonzero(d == 1)


def test_degradar_huecos_ponderados_por_area_y_deterministas():
    p = partido_sintetico(t=4200)
    p.balon[:, 0] = np.linspace(0.05, 0.95, p.n_frames)
    p.balon[:, 1] = 0.5
    base = Degradacion(balon_ausente=0.2, hueco_balon_mediana=20, hueco_balon_sigma=0.1,
                       fragmento_mediana=300, semilla=4)
    area = Degradacion(balon_ausente=0.2, hueco_balon_mediana=20, hueco_balon_sigma=0.1,
                       fragmento_mediana=300, hueco_peso_area=8, semilla=4)
    a = degradar(p, area)
    b = degradar(p, area)
    np.testing.assert_array_equal(a.balon, b.balon)
    ini_base = _inicios_huecos(degradar(p, base))
    ini_area = _inicios_huecos(a)
    def frac_area(indices):
        x, y = p.balon[indices, 0], p.balon[indices, 1]
        return (((x < 0.17) | (x > 0.83)) & (y > 0.2) & (y < 0.8)).mean()
    assert frac_area(ini_area) > frac_area(ini_base)


def test_baseline_sin_balon_es_D():
    p = partido_sintetico()
    p.balon[:100] = np.nan
    assert (baseline.suavizar(baseline.mas_cercano(p), 0)[:100] == D).all()


ATD = os.path.join(RAIZ, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@pytest.mark.skipif(not os.path.exists(ATD), reason="sin datos del reto")
@pytest.mark.parametrize("ventana", [0, 25, 50, 125])
def test_port_baseline_identico_al_oficial(ventana):
    """El port sobre la representación común da el mismo CSV que el baseline oficial."""
    import sys
    sys.path.insert(0, os.path.join(RAIZ, "evaluation"))
    import possession_metrics as pm
    from mtah.fuentes import reto
    p = reto.cargar(gt_publico=False)
    nombre = "baseline_raw" if ventana == 0 else "baseline_smooth{}".format(ventana)
    oficial = pm.label_codes(pm.read_label_csv(
        os.path.join(RAIZ, "evaluation", "results", nombre + ".csv")))
    nuestro = baseline.predictor(ventana)(p)
    assert np.array_equal(nuestro, oficial), int((nuestro != oficial).sum())
