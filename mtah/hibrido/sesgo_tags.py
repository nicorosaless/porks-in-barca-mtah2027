"""Smart Tagging -> potenciales que se suman al logp [S, 3] (H, A, D) antes de Viterbi.

Términos blandos, por paso (`fps_paso` pasos por segundo; un paso s cubre los frames
[s * k, s * k + k) con k = 25 / fps_paso, como en decodificar.py):

    ataque      fase de ataque con Team          +α al Team
    defensa     DEFENDING* y DEFENSIVE TRANSITION +α al rival del Team (`defensa_es_rival`)
    fase        cualquiera de las dos con Team    −γ a D
    saque       SET PIECES                        +β a D en `ventana_antes_s` antes del inicio
                                                  del tag; +β al Team en `ventana_despues_s`
                                                  tras el fin del tag (con N/A, solo lo de D)

Dentro del tag de un saque no se dice nada: el tag es una ventana de ~5 s alrededor de la
reanudación (starter/load_smart_tagging.py) y no se sabe en qué punto cae. Los solapes no
suman: cada término es un indicador por paso. α, β y γ, y la suma blanda de cada celda, se
recortan a ±`tope` nats.

DATA.md no dice si el Team de un DEFENDING* es el que defiende o el que ataca. Por defecto se
toma el que defiende (+α al rival); `defensa_es_rival=False` lo invierte.

Reglas duras (opcionales, `reglas_duras`): H y A a −`fuerza_dura` (D forzado) en
    - el tramo evaluable antes de cada `Kick Off (Start)`: con `seccion`, desde el inicio del
      bloque de su parte; sin ella, desde el fin de la instancia anterior y como mucho
      `previo_saque_inicial_s` antes. FAQ.md: los 2.934 frames evaluables previos al saque de
      la segunda parte son D;
    - entre el fin de cada `GOALS` y el siguiente `Kick Off (After Goal)` de la misma parte
      (como mucho `max_gol_saque_s` después).
Los pasos que solo tocan en parte un tag se quedan fuera de las reglas duras.

Con α = β = γ = 0 y sin reglas duras devuelve ceros.
"""
from dataclasses import dataclass

import numpy as np

from mtah.fuentes.smart_tagging import SIN_EQUIPO
from mtah.partido import A, D, FPS, H

ATAQUE = frozenset({"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL"})
DEFENSA = frozenset({"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"})
SAQUE = "SET PIECES"
GOL = "GOALS"
SAQUE_INICIAL = "kick off (start)"
SAQUE_TRAS_GOL = "kick off (after goal)"


@dataclass(frozen=True)
class ParamsSesgo:
    alfa: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    tope: float = 2.0
    ventana_antes_s: float = 2.0
    ventana_despues_s: float = 2.0
    defensa_es_rival: bool = True
    reglas_duras: bool = False
    fuerza_dura: float = 50.0
    previo_saque_inicial_s: float = 150.0
    max_gol_saque_s: float = 300.0


def _code(i):
    return " ".join(i.code.split()).upper()


def _tipo(i):
    return " ".join((i.type or "").split()).lower()


def _es_defensa(code):
    return code in DEFENSA or code.startswith("DEFENDING")


def _pasos(ini, fin, k, n_pasos, cubrir=True):
    """Frames [ini, fin) -> pasos. `cubrir`: todo paso que toque; si no, solo los enteros."""
    if cubrir:
        a, b = ini // k, -(-fin // k)
    else:
        a, b = -(-ini // k), fin // k
    return slice(max(0, min(a, n_pasos)), max(0, min(b, n_pasos)))


def _bloque(seccion, f):
    """Primer frame del bloque contiguo de `seccion` que contiene f; None si f no es evaluable."""
    if not 0 <= f < len(seccion) or seccion[f] == 0:
        return None
    previos = np.flatnonzero(seccion[:f] != seccion[f])
    return int(previos[-1]) + 1 if len(previos) else 0


def _duras(instancias, n_pasos, k, p, seccion):
    out = np.zeros(n_pasos, bool)
    orden = sorted(instancias, key=lambda i: (i.inicio_frame, i.fin_frame))
    for n, i in enumerate(orden):
        if _code(i) != SAQUE or _tipo(i) != SAQUE_INICIAL:
            continue
        if seccion is not None:
            desde = _bloque(seccion, i.inicio_frame)
            if desde is None:
                continue
        else:
            antes = [j.fin_frame for j in orden[:n] if j.fin_frame <= i.inicio_frame]
            desde = max([i.inicio_frame - int(round(p.previo_saque_inicial_s * FPS))] + antes)
        out[_pasos(desde, i.inicio_frame, k, n_pasos, cubrir=False)] = True
    tope = int(round(p.max_gol_saque_s * FPS))
    for g in orden:
        if _code(g) != GOL:
            continue
        saques = [i for i in orden if _code(i) == SAQUE and _tipo(i) == SAQUE_TRAS_GOL
                  and g.fin_frame <= i.inicio_frame <= g.fin_frame + tope
                  and (g.half == 0 or i.half == 0 or i.half == g.half)]
        if saques:
            out[_pasos(g.fin_frame, saques[0].inicio_frame, k, n_pasos, cubrir=False)] = True
    return out


def sesgo(instancias, n_pasos, fps_paso=5, params=ParamsSesgo(), seccion=None):
    """Potenciales [n_pasos, 3] (H, A, D) para sumar al logp por paso antes de Viterbi.

    `seccion` (opcional, int8 por frame como `Partido.seccion`) solo afina la regla dura del saque
    inicial; en datos externos conviene pasarla, porque las partes van pegadas.
    """
    if FPS % fps_paso:
        raise ValueError("fps_paso debe dividir {}".format(FPS))
    k = FPS // fps_paso
    p = params
    a_, b_, g_ = (float(np.clip(v, -p.tope, p.tope)) for v in (p.alfa, p.beta, p.gamma))
    equipo = np.zeros((n_pasos, 2), bool)        # evidencia de fase por equipo
    fase = np.zeros(n_pasos, bool)
    d_saque = np.zeros(n_pasos, bool)
    tras_saque = np.zeros((n_pasos, 2), bool)
    antes = int(round(p.ventana_antes_s * FPS))
    despues = int(round(p.ventana_despues_s * FPS))
    for i in instancias:
        code = _code(i)
        if code == SAQUE:
            d_saque[_pasos(i.inicio_frame - antes, i.inicio_frame, k, n_pasos)] = True
            if i.team in (H, A):
                tras_saque[_pasos(i.fin_frame, i.fin_frame + despues, k, n_pasos), i.team] = True
            continue
        if i.team == SIN_EQUIPO or i.team not in (H, A):
            continue
        if code in ATAQUE:
            lado = i.team
        elif _es_defensa(code):
            lado = 1 - i.team if p.defensa_es_rival else i.team
        else:
            continue
        s = _pasos(i.inicio_frame, i.fin_frame, k, n_pasos)
        equipo[s, lado] = True
        fase[s] = True

    out = np.zeros((n_pasos, 3), np.float64)
    out[:, [H, A]] += a_ * equipo + b_ * tras_saque
    out[:, D] += b_ * d_saque - g_ * fase
    np.clip(out, -p.tope, p.tope, out=out)
    if p.reglas_duras:
        duro = _duras(instancias, n_pasos, k, p, seccion)
        out[duro, H] -= p.fuerza_dura
        out[duro, A] -= p.fuerza_dura
    return out
