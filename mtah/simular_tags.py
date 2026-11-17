"""Instancias tipo Smart Tagging simuladas en datos externos, para ajustar α, β y γ sin el reto.

Salen en la misma estructura que mtah/fuentes/smart_tagging.py (`Instancia`, frames de 25 fps
del `Partido` que construye el loader de cada fuente, partes separadas por 25 frames).

Fuentes
-------
- SkillCorner: fases de `phases_of_play.csv` -> códigos de fase; reanudaciones y goles de los
  `player_possession` de `dynamic_events.csv` (`game_interruption_before/after`).
    en posesión   build_up BUILD UP · create PROGRESSION · finish CREATING CHANCES ·
                  direct LONG BALL · quick_break/transition ATTACKING TRANSITION;
                  además BALL IN FINAL THIRD si acaba en el último tercio y BALL IN THE BOX y
                  PLAYERS IN THE BOX si acaba en el área. chaotic y set_play no dan fase.
    sin posesión  high_block DEFENDING IN ATTACKING THIRD · medium_block ... MIDDLE THIRD ·
                  low_block ... DEFENSIVE THIRD · defending_(quick_break|transition)
                  DEFENSIVE TRANSITION. Team = el que defiende (`team_defensa="defiende"`, la
                  convención por defecto de sesgo_tags) o el que ataca.
    saques        throw_in/free_kick/goal_kick/corner `_for` = saca el equipo de la posesión,
                  `_against` = el rival. Ventana [recepción − 3 s, recepción + 2 s). Banda: N/A.
- Metrica sample: eventos `SET PIECE` (ventana de ±2,5 s alrededor del saque) y goles (`SHOT`
  con GOAL). No hay fases. No hay GOAL KICK en los subtipos.
- IDSSE: el loader no expone eventos y en $MTAH_EXTERNAL_DIR/idsse no están los DFL_03_02. No
  se simula.

Saque inicial y GOALS van siempre con Team N/A (no se sabe qué pone el fichero real).

Ruido contra la fuga (por defecto)
----------------------------------
En SkillCorner las D del GT salen de las mismas `phases_of_play`; en Metrica, de los mismos
`SET PIECE`. Sin ruido, los bordes de las instancias son los del GT. Por eso:
    - cobertura: por código de fase, como mucho n_real(código) × minutos / MINUTOS_REALES
      instancias al azar (tabla de DATA.md y frames evaluables publicados);
    - recall por tipo de saque: banda 0,5 · falta 0,25 · puerta 0,65 · córner 0,9;
    - volteo de Team con prob. 0,2;
    - jitter independiente de inicio y fin ~U(−2 s, +2 s).
"""
import csv
import json
import os
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mtah.fuentes import metrica as fm
from mtah.fuentes.smart_tagging import SIN_EQUIPO, Instancia
from mtah.partido import A, FPS, H
from mtah.rutas import cache_dir, external_dir

SEPARACION = 25

# DATA.md, tabla de Smart Tagging (cuentas publicadas del fichero real, 638 instancias)
REAL_POR_CODIGO = {
    "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
    "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
    "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
    "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
    "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
}
# Frames evaluables publicados (DATA.md, FAQ.md): 158.857 a 25 fps = 105,9 min
MINUTOS_REALES = 158_857 / FPS / 60

BANDA, FALTA, PUERTA, CORNER = "Throw-in", "Free Kick", "Goal Kick", "Corner Kick"
SAQUE_INICIAL, SAQUE_GOL = "Kick Off (Start)", "Kick Off (After Goal)"
SAQUE, GOL = "SET PIECES", "GOALS"

_FASE_ATAQUE = {"build_up": "BUILD UP", "create": "PROGRESSION", "finish": "CREATING CHANCES",
                "direct": "LONG BALL", "quick_break": "ATTACKING TRANSITION",
                "transition": "ATTACKING TRANSITION"}
_FASE_DEFENSA = {"high_block": "DEFENDING IN ATTACKING THIRD",
                 "medium_block": "DEFENDING IN MIDDLE THIRD",
                 "low_block": "DEFENDING IN DEFENSIVE THIRD",
                 "defending_transition": "DEFENSIVE TRANSITION",
                 "defending_quick_break": "DEFENSIVE TRANSITION"}
_REANUDA_SC = {"throw_in": BANDA, "free_kick": FALTA, "goal_kick": PUERTA, "corner": CORNER}


@dataclass(frozen=True)
class Ruido:
    jitter_s: float = 2.0
    volteo: float = 0.2
    cobertura: Optional[float] = 1.0      # factor sobre la cobertura real; None = no descarta
    recall_banda: float = 0.5
    recall_falta: float = 0.25
    recall_puerta: float = 0.65
    recall_corner: float = 0.9
    recall_saque_centro: float = 1.0
    recall_gol: float = 1.0
    team_defensa: str = "defiende"        # Team de DEFENDING*: "defiende" o "ataca"


SIN_RUIDO = Ruido(jitter_s=0.0, volteo=0.0, cobertura=None, recall_banda=1.0, recall_falta=1.0,
                  recall_puerta=1.0, recall_corner=1.0)


def _s(x):
    return int(round(x * FPS))


def _ventana(centro, antes_s, despues_s, team, tipo, code=SAQUE, half=0):
    return Instancia(centro - _s(antes_s), centro + _s(despues_s), code, team, tipo, half)


def _dedup(momentos, s=30):
    out = []
    for m in sorted(momentos):
        if not out or m - out[-1] > _s(s):
            out.append(m)
    return out


# ---------------------------------------------------------------- SkillCorner


def _bordes_skillcorner(mid, meta):
    """({periodo: (frame10_inicio, off25, n25)}, n_frames). Longitudes de la caché si existe."""
    cache = os.path.join(cache_dir(), "skillcorner_{}.npz".format(mid))
    sec = np.load(cache)["seccion"] if os.path.exists(cache) else None
    bordes, off = {}, 0
    for mp in sorted(meta["match_periods"], key=lambda m: m["period"]):
        per = int(mp["period"])
        if per not in (1, 2):
            continue
        a = int(mp["start_frame"])
        if sec is not None:
            idx = np.flatnonzero(sec == per)
            off, n = int(idx[0]), len(idx)
        else:
            n = (int(mp["end_frame"]) - a) * 25 // 10 + 1
        bordes[per] = (a, off, n)
        off += n + SEPARACION
    return bordes, (len(sec) if sec is not None else off)


def _candidatos_skillcorner(mid, team_defensa):
    ruta = os.path.join(external_dir(), "skillcorner", "matches", str(mid))
    with open(os.path.join(ruta, "{}_match.json".format(mid))) as f:
        meta = json.load(f)
    bordes, n_frames = _bordes_skillcorner(mid, meta)
    lado = {int(meta["home_team"]["id"]): H, int(meta["away_team"]["id"]): A}

    def a25(f10, per):
        a, off, n = bordes[per]
        return off + min(max(int(np.floor((f10 - a) * 2.5 + 0.5)), 0), n)

    def per_de(row):
        p = int(float(row.get("period") or 0))
        return p if p in bordes else None

    out = [_ventana(off, 0, 5.0, SIN_EQUIPO, SAQUE_INICIAL, half=p)
           for p, (_, off, _) in sorted(bordes.items())]
    with open(os.path.join(ruta, "{}_phases_of_play.csv".format(mid)), newline="") as f:
        for row in csv.DictReader(f):
            per, t = per_de(row), row.get("team_in_possession_id") or ""
            ataca = lado.get(int(float(t))) if t else None
            if per is None or ataca is None:
                continue
            ini = a25(int(row["frame_start"]), per)
            fin = a25(int(row["frame_end"]) + 1, per)
            if fin <= ini:
                continue
            codes = [(_FASE_ATAQUE.get(row["team_in_possession_phase_type"]), ataca)]
            if row.get("third_end") == "attacking_third":
                codes.append(("BALL IN FINAL THIRD", ataca))
            if row.get("penalty_area_end") == "True":
                codes += [("BALL IN THE BOX", ataca), ("PLAYERS IN THE BOX", ataca)]
            defiende = 1 - ataca if team_defensa == "defiende" else ataca
            codes.append((_FASE_DEFENSA.get(row["team_out_of_possession_phase_type"]), defiende))
            out += [Instancia(ini, fin, c, e, "", per) for c, e in codes if c]

    goles, saques_gol = [], []
    with open(os.path.join(ruta, "{}_dynamic_events.csv".format(mid)), newline="") as f:
        for row in csv.DictReader(f):
            per = per_de(row)
            if row.get("event_type") != "player_possession" or per is None:
                continue
            equipo = lado.get(int(float(row["team_id"]))) if row.get("team_id") else None
            antes = row.get("game_interruption_before") or ""
            despues = row.get("game_interruption_after") or ""
            t0 = a25(int(row["frame_start"]), per)
            if antes in ("goal_for", "goal_against"):   # ojo: no goal_kick_*
                saques_gol.append((t0, per))
            for clave, tipo in _REANUDA_SC.items():
                if antes.startswith(clave + "_") and equipo is not None:
                    saca = equipo if antes.endswith("_for") else 1 - equipo
                    out.append(_ventana(t0, 3.0, 2.0, SIN_EQUIPO if tipo == BANDA else saca,
                                        tipo, half=per))
            if despues in ("goal_for", "goal_against"):
                goles.append((a25(int(row["frame_end"]) + 1, per), per))
    for momentos, crea in ((saques_gol, lambda t, p: _ventana(t, 3.0, 2.0, SIN_EQUIPO, SAQUE_GOL,
                                                                half=p)),
                           (goles, lambda t, p: _ventana(t, 8.0, 4.0, SIN_EQUIPO, "", GOL, p))):
        for p in (1, 2):
            out += [crea(t, p) for t in _dedup(t for t, q in momentos if q == p)]
    return out, n_frames


# ---------------------------------------------------------------- Metrica


def _bordes_metrica(sg):
    """{periodo: (primer frame del fichero, off25, n)}, n_frames. Igual que metrica._convertir."""
    if sg == 3:
        rangos = {1: (1, 69661), 2: (69662, 143761)}     # los de metrica._leer_tracking_sg3
        primeros = {p: (lo, hi - lo + 1) for p, (lo, hi) in rangos.items()}
    else:
        primeros = {}
        ruta = fm._ruta(sg, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(sg))
        with open(ruta, newline="") as f:
            filas = csv.reader(f)
            for _ in range(3):
                next(filas)
            for row in filas:
                p, fr = int(row[0]), int(row[1])
                lo, n = primeros.get(p, (fr, 0))
                primeros[p] = (lo, n + 1)
    bordes, off = {}, 0
    for p in (1, 2):
        lo, n = primeros[p]
        bordes[p] = (lo, off, n)
        off += n + SEPARACION
    return bordes, off


def _candidatos_metrica(id_):
    sg = int(str(id_)[-1])
    bordes, n_frames = _bordes_metrica(sg)
    if sg in (1, 2):
        eventos = fm._eventos_csv(fm._ruta(sg, "Sample_Game_{}_RawEventsData.csv".format(sg)))
    else:
        eventos = fm._eventos_json(fm._ruta(3, "Sample_Game_3_events.json"))
    out, con_saque = [], set()
    for e in sorted(eventos, key=lambda e: (e.periodo, e.inicio)):
        if e.periodo not in bordes:
            continue
        lo, off, n = bordes[e.periodo]
        ini = off + min(max(e.inicio - lo, 0), n)
        fin = off + min(max(e.fin - lo, 0), n)
        tipo, sub = e.tipo.upper(), e.subtipos.upper()
        team = H if e.equipo == 0 else A
        if tipo == "SET PIECE" and "RETAKEN" not in sub:
            if "KICK OFF" in sub:
                t = SAQUE_GOL if e.periodo in con_saque else SAQUE_INICIAL
                con_saque.add(e.periodo)
                out.append(_ventana(ini, 2.5, 2.5, SIN_EQUIPO, t, half=e.periodo))
                continue
            t = (BANDA if "THROW IN" in sub else CORNER if "CORNER" in sub
                 else PUERTA if "GOAL KICK" in sub else FALTA if "FREE KICK" in sub else None)
            if t:
                out.append(_ventana(ini, 2.5, 2.5, SIN_EQUIPO if t == BANDA else team, t,
                                    half=e.periodo))
        elif tipo == "SHOT" and "GOAL" in sub and "WOODWORK" not in sub:
            out.append(_ventana(fin, 8.0, 4.0, SIN_EQUIPO, "", GOL, e.periodo))
    return out, n_frames


# ---------------------------------------------------------------- común


def candidatos(fuente, partido, team_defensa="defiende"):
    """(instancias sin ruido, n_frames del `Partido`). Bordes = los de la fuente, sin tocar."""
    if fuente == "skillcorner":
        return _candidatos_skillcorner(str(partido), team_defensa)
    if fuente == "metrica":
        return _candidatos_metrica(partido)
    if fuente == "idsse":
        raise NotImplementedError("IDSSE: el loader no expone eventos y no están los DFL_03_02 "
                                  "en $MTAH_EXTERNAL_DIR/idsse; no hay saques que simular")
    raise ValueError("fuente desconocida: {}".format(fuente))


def _recall(ruido, i):
    if i.code == GOL:
        return ruido.recall_gol
    return {BANDA: ruido.recall_banda, FALTA: ruido.recall_falta, PUERTA: ruido.recall_puerta,
            CORNER: ruido.recall_corner, SAQUE_INICIAL: ruido.recall_saque_centro,
            SAQUE_GOL: ruido.recall_saque_centro}.get(i.type, 1.0)


def aplicar_ruido(cands, n_frames, rng, ruido=Ruido()):
    """Cobertura y recall, volteo de Team y jitter, en ese orden. Determinista dado `rng`."""
    cands = sorted(set(cands))
    vivos = [i for i in cands if i.code not in (SAQUE, GOL)]
    # minutos del partido externo: sus dos partes, sin las separaciones de 25 frames
    minutos = max(n_frames - 2 * SEPARACION, 0) / FPS / 60
    quedan = []
    for code in sorted({i.code for i in vivos}):
        grupo = [i for i in vivos if i.code == code]
        if ruido.cobertura is None:
            quedan += grupo
            continue
        m = int(round(REAL_POR_CODIGO.get(code, 0) * minutos / MINUTOS_REALES * ruido.cobertura))
        m = min(m, len(grupo))
        quedan += [grupo[j] for j in np.sort(rng.choice(len(grupo), m, replace=False))]
    quedan += [i for i in cands if i.code in (SAQUE, GOL) and rng.random() < _recall(ruido, i)]

    j = ruido.jitter_s * FPS
    out = []
    for i in sorted(quedan):
        team = i.team
        if team in (H, A) and rng.random() < ruido.volteo:
            team = 1 - team
        a = min(max(i.inicio_frame + int(round(rng.uniform(-j, j))), 0), n_frames - 1)
        b = min(max(i.fin_frame + int(round(rng.uniform(-j, j))), a + 1), n_frames)
        out.append(i._replace(inicio_frame=a, fin_frame=b, team=team))
    return sorted(out)


def simular(fuente, partido, semilla=0, ruido=Ruido()):
    """Instancias simuladas con ruido para un partido externo. Mismo resultado con la misma
    semilla; la semilla se mezcla con la fuente y el partido para no repetir el ruido."""
    cands, n_frames = candidatos(fuente, partido, ruido.team_defensa)
    rng = np.random.default_rng([int(semilla), zlib.crc32("{}:{}".format(fuente, partido).encode())])
    return aplicar_ruido(cands, n_frames, rng, ruido)
