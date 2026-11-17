"""Registro de predictores. Un predictor es una función Partido -> códigos int8 [T] (H/A/D/X).

Para añadir un método nuevo: crea un módulo en este paquete y regístralo en PREDICTORES.
scripts/evaluar_externo.py los evalúa todos en las mismas condiciones.
"""
from mtah.predictores import baseline, hibrido

PREDICTORES = {
    "baseline_raw": baseline.predictor(0),
    "baseline_smooth25": baseline.predictor(25),
    "baseline_smooth50": baseline.predictor(50),
    "baseline_smooth125": baseline.predictor(125),
}
# modelos/hibrido/<nombre>.json, con parámetros ajustados en IDSSE
PREDICTORES.update({"hibrido_" + n: hibrido.predictor(n) for n in hibrido.disponibles()})
