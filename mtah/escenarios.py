"""Escenarios de degradación usados para entrenar, evaluar y afinar el híbrido."""
from dataclasses import replace

from mtah.degradar import Degradacion

PESIMISTA = dict(congelado_ausente=0.0, congelado_fuera_ausente=0.0)
V3 = dict(
    hueco_peso_cerca=2.0,
    hueco_peso_area=0.0,
    hueco_peso_fuera=0.0,
    congelado_fuera_ausente=0.7,
    hueco_minimo_largo=125,
    hueco_corto_prob=0.1,
)
# E5: v3 + parámetros no publicados sorteados por semilla + defectos de DATA.md (tasas a priori)
V4 = dict(
    V3,
    aleatorizar=True,
    hueco_peso_rapido=2.0,
    duplicados_por_min=0.5,
    balon_falso_por_min=0.3,
    equipo_erroneo=0.03,
)
ESCENARIOS = {
    "v2": {},
    "pes": PESIMISTA,
    "v3": V3,
    "v3pes": {**V3, **PESIMISTA},
    "v4": V4,
}
SUFIJO = {
    "v2": "",
    "pes": "_pes",
    "v3": "_v3",
    "v3pes": "_v3pes",
    "v4": "_v4",
}


def cfg(nombre, semilla=0):
    return replace(Degradacion(semilla=semilla), **ESCENARIOS[nombre])
