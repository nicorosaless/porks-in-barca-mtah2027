import numpy as np
import torch

from mtah.hibrido.modelo import TRANSFORMACIONES, Red, predecir, predecir_tta, transformar
from mtah.hibrido.rasgos import FB, FJ, KV


def _r(s=60, semilla=0):
    rng = np.random.default_rng(semilla)
    jug = rng.normal(size=(s, KV, FJ)).astype(np.float32)
    jug[..., 1] = np.sign(rng.normal(size=(s, KV)))  # local = +1/-1
    seccion = np.zeros(s, np.int8)
    seccion[5:55] = 1
    return {"balon": rng.normal(size=(s, FB)).astype(np.float32), "jug": jug, "seccion": seccion}


def _red():
    torch.manual_seed(0)
    return Red(d=16, dt=16).eval()


def test_identidad_igual_que_predecir():
    red, r = _red(), _r()
    np.testing.assert_allclose(predecir_tta(red, r, transformaciones=[(0, 0, 0)]), predecir(red, r), atol=1e-6)


def test_tta_normaliza_y_es_invariante_al_cambio_de_equipos():
    red, r = _red(), _r()
    lp = predecir_tta(red, r)
    assert len(TRANSFORMACIONES) == 8
    np.testing.assert_allclose(np.exp(lp).sum(1), 1, atol=1e-5)
    lp_sw = predecir_tta(red, transformar(r, 0, 0, 1))
    np.testing.assert_allclose(lp_sw[:, [1, 0, 2]], lp, atol=1e-5)
    lp_fx = predecir_tta(red, transformar(r, 1, 0, 0))
    np.testing.assert_allclose(lp_fx, lp, atol=1e-5)
