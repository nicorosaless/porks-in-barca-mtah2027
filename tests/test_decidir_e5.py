import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "hibrido"))
from decidir_e5 import decidir  # noqa: E402


def test_pasa_con_mejora_clara_y_si_estable():
    r = decidir([0.60, 0.61, 0.62], [0.65, 0.66, 0.67], [0.70] * 3, [0.69] * 3)
    assert r["pasa"] and abs(r["umbral"] - 0.02) < 1e-9


def test_no_pasa_si_la_mejora_no_supera_dos_sigma():
    r = decidir([0.55, 0.60, 0.65], [0.58, 0.63, 0.68], [0.70] * 3, [0.70] * 3)
    assert not r["pasa"] and r["umbral"] > 0.09


def test_no_pasa_si_si_cae_mas_de_002():
    r = decidir([0.60] * 3, [0.70] * 3, [0.70] * 3, [0.67] * 3)
    assert not r["pasa"]
