import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "hibrido"))
from ensamblar import mezclar  # noqa: E402


def test_mezcla_de_probabilidades():
    a = np.log(np.array([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]))
    b = np.log(np.array([[0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]))
    p = np.exp(mezclar([a, b], [0.5, 0.5]))
    np.testing.assert_allclose(p, [[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]], atol=1e-12)
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_predictor_mezcla_igual_que_ensamblar():
    from mtah.predictores.hibrido import mezclar as mezclar_pred
    rng = np.random.default_rng(0)
    a, b = (np.log(rng.dirichlet(np.ones(3), 50)) for _ in range(2))
    np.testing.assert_allclose(mezclar_pred([a, b], [0.5, 0.5]), mezclar([a, b], [0.5, 0.5]))
