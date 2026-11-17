import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "hibrido"))
from entrenar import _extra_fuente  # noqa: E402


def test_huecos_estilo_solo_en_fuentes_estilo():
    extra = {"congelado_ausente": 0.35, "balon_ausente_estilo": 0.6}
    assert _extra_fuente("skillcorner:1", extra) == {
        "congelado_ausente": 0.35, "balon_ausente": 0.6, "quitar_congelado": False}
    assert _extra_fuente("idsse:J03WMX", extra) == {"congelado_ausente": 0.35}
    assert extra["balon_ausente_estilo"] == 0.6  # no muta la entrada


def test_sin_huecos_estilo_igual_que_antes():
    assert _extra_fuente("metrica:SG1", {}) == {"balon_ausente": 0.0, "quitar_congelado": False}
    assert _extra_fuente("idsse:J03WMX", {}) == {}
