import importlib.util
import os
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from mtah.degradar import Degradacion, degradar
from mtah.escenarios import ESCENARIOS, cfg
from mtah.partido import Partido

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _partido(t=3000, semilla=0):
    rng = np.random.default_rng(semilla)
    balon = np.clip(0.5 + np.cumsum(rng.normal(0, 0.004, (t, 2)), 0), 0.02, 0.98).astype(np.float32)
    balon[400:440] = balon[400]  # tramo congelado de DFL
    jug = np.clip(rng.uniform(0.1, 0.9, (1, 22, 2)) + np.cumsum(rng.normal(0, 0.002, (t, 22, 2)), 0),
                  0, 1).astype(np.float32)
    seccion = np.zeros(t, np.int8)
    seccion[100:1500] = 1
    seccion[1600:2900] = 2
    return Partido(nombre="sint", campo=(105.0, 68.0), balon=balon, jugadores=jug,
                   equipo_slot=np.array([0] * 11 + [1] * 11, np.int8), seccion=seccion)


def _degradar_head():
    """El degradar.py del último commit, para fijar que v2 y v3 no cambian."""
    try:
        src = subprocess.run(["git", "show", "HEAD:mtah/degradar.py"], cwd=RAIZ, check=True,
                             capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("sin git")
    ruta = os.path.join(RAIZ, "tests", "__pycache__", "degradar_head.py")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    open(ruta, "w").write(src)
    spec = importlib.util.spec_from_file_location("degradar_head", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("esc", ["v2", "pes", "v3", "v3pes"])
def test_escenarios_anteriores_identicos_al_commit(esc):
    head = _degradar_head()
    p = _partido()
    nuevo = degradar(p, cfg(esc, semilla=3))
    viejo = head.degradar(p, replace(head.Degradacion(semilla=3), **ESCENARIOS[esc]))
    np.testing.assert_array_equal(nuevo.balon, viejo.balon)
    np.testing.assert_array_equal(nuevo.jugadores, viejo.jugadores)
    np.testing.assert_array_equal(nuevo.equipo_slot, viejo.equipo_slot)


def test_v4_determinista_y_distinto_de_v3():
    p = _partido()
    a = degradar(p, cfg("v4", semilla=5))
    b = degradar(p, cfg("v4", semilla=5))
    np.testing.assert_array_equal(a.balon, b.balon)
    np.testing.assert_array_equal(a.jugadores, b.jugadores)
    c = degradar(p, cfg("v3", semilla=5))
    assert a.jugadores.shape != c.jugadores.shape or not np.array_equal(a.jugadores, c.jugadores, equal_nan=True)


def test_aleatorizar_cambia_parametros_calibrados_por_semilla_y_respeta_publicados():
    from mtah.degradar import aleatorizada
    base = Degradacion(aleatorizar=True)
    medianas = {aleatorizada(replace(base, semilla=s)).hueco_balon_mediana for s in range(20)}
    assert len(medianas) == 20
    for s in range(20):
        d = aleatorizada(replace(base, semilla=s))
        assert d.balon_ausente == base.balon_ausente  # 40,7 %: publicado en DATA.md
        assert d.slots_por_equipo == base.slots_por_equipo  # 46 slots: publicado
        assert 120 <= d.hueco_balon_mediana <= 400
    assert aleatorizada(Degradacion(semilla=1)) == Degradacion(semilla=1)


def test_duplicados_superan_22_visibles_a_veces():
    p = _partido()
    d = Degradacion(semilla=2, duplicados_por_min=6.0, fragmento_mediana=1e6, hueco_jugador_media=1)
    q = degradar(p, d)
    vis = (~np.isnan(q.jugadores[:, :, 0])).sum(1)
    assert vis.max() > 22


def test_balon_falso_mueve_el_balon_observado():
    p = _partido()
    base = Degradacion(semilla=4, ruido_balon_m=0.0, quitar_congelado=False, balon_ausente=0.0)
    limpio = degradar(p, base).balon
    falso = degradar(p, replace(base, balon_falso_por_min=5.0)).balon
    obs = ~np.isnan(limpio[:, 0]) & ~np.isnan(falso[:, 0])
    salto = np.linalg.norm((falso - limpio)[obs] * np.array([105, 68]), axis=1)
    assert (salto > 3).mean() > 0.005


def test_equipo_erroneo_mete_jugadores_en_el_otro_equipo():
    p = _partido()
    d = Degradacion(semilla=1, equipo_erroneo=0.5, fragmento_mediana=200, fragmento_sigma=0.1)
    q = degradar(p, d)
    # el jugador 0 (local) acaba en algún slot visitante en algún momento
    vis_visit = q.jugadores[:, q.equipo_slot == 1]
    pos0 = p.jugadores[:, 0]
    iguales = np.isclose(vis_visit, pos0[:, None, :], atol=0.02).all(-1).any(1)
    assert iguales.any()


def test_huecos_prefieren_balon_rapido():
    p = _partido()
    v = np.r_[0, np.linalg.norm(np.diff(p.balon, axis=0) * np.array([105, 68]), axis=1) * 25]
    rapido = v > np.quantile(v, 0.8)
    def frac(peso):
        tot = 0.0
        for s in range(4):
            q = degradar(p, Degradacion(semilla=s, quitar_congelado=False, hueco_peso_rapido=peso,
                                        hueco_balon_mediana=10, hueco_balon_sigma=0.2))
            falta = np.isnan(q.balon[:, 0]) & (p.seccion > 0)
            tot += falta[rapido].mean() / max(falta[~rapido].mean(), 1e-9)
        return tot / 4
    assert frac(5.0) > frac(0.0) * 1.3
