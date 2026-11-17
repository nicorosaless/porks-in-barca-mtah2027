"""Smart Tagging: parser, sesgo para Viterbi y simulador. Todo con datos sintéticos escritos aquí."""
import csv
import json
import os

import numpy as np
import pytest

from mtah.fuentes.smart_tagging import SIN_EQUIPO, Instancia, agregados, parsear
from mtah.hibrido.sesgo_tags import ParamsSesgo, sesgo
from mtah.partido import A, D, H
from mtah import simular_tags

XML = """<?xml version="1.0" encoding="utf-8"?>
<file><ALL_INSTANCES>
{}
</ALL_INSTANCES></file>"""


def _inst(i, start, end, code, **tags):
    labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                     for g, t in tags.items())
    return ("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>{}</instance>"
            .format(i, start, end, code, labels))


@pytest.fixture
def xml(tmp_path):
    cuerpo = "\n".join([
        _inst(1, 10.0, 12.02, "SET PIECES", Team="N/A", Type="Throw-in", Half="1st Half"),
        _inst(2, 12.5, 20.0, "BUILD UP", Team="Barcelona", Half="1st Half"),
        _inst(3, 21.0, 25.0, "DEFENDING IN MIDDLE THIRD", Team="atletic club ", Half="2nd Half"),
        _inst(4, 30.0, 31.0, "goals"),
    ])
    ruta = tmp_path / "st.xml"
    ruta.write_text(XML.format(cuerpo))
    return str(ruta)


# ---------- parser


def test_parser_frames_equipo_tipo_y_parte(xml):
    out = parsear(xml, ["Barcelona", "Atletic Club"])
    assert out[0] == Instancia(250, 300, "SET PIECES", SIN_EQUIPO, "Throw-in", 1)  # 12.02*25=300.5
    assert out[1] == Instancia(312, 500, "BUILD UP", H, "", 1)                      # 12.5*25=312.5
    assert out[2].team == A and out[2].half == 2      # mayúsculas y espacios no importan
    assert out[3].code == "GOALS" and out[3].team == SIN_EQUIPO and out[3].half == 0


def test_parser_falla_con_equipo_desconocido(xml):
    with pytest.raises(ValueError):
        parsear(xml, ["Barcelona", "Athletic Bilbao"])


def test_agregados_solo_cuentas(xml):
    ag = agregados(parsear(xml, ["Barcelona", "Atletic Club"]))
    assert ag == {"por_codigo": {"BUILD UP": 1, "DEFENDING IN MIDDLE THIRD": 1, "GOALS": 1,
                                 "SET PIECES": 1},
                  "n_na": 2, "H": 1, "A": 1}


# ---------- sesgo

K = 5  # frames por paso a 5 fps


def _i(ini, fin, code, team=SIN_EQUIPO, tipo="", half=1):
    return Instancia(ini * K, fin * K, code, team, tipo, half)


def test_sesgo_ceros_por_defecto():
    inst = [_i(2, 6, "BUILD UP", H), _i(8, 9, "SET PIECES", A, "Free Kick"),
            _i(0, 1, "SET PIECES", tipo="Kick Off (Start)")]
    out = sesgo(inst, 20)
    assert out.shape == (20, 3) and not out.any()


def test_sesgo_ataque_defensa_y_gamma():
    p = ParamsSesgo(alfa=1.0, gamma=0.5)
    out = sesgo([_i(2, 4, "PROGRESSION", H), _i(6, 8, "Defending in Defensive Third", H)], 10,
                params=p)
    assert np.allclose(out[2:4], [1.0, 0.0, -0.5])     # ataque: +α al Team, −γ a D
    assert np.allclose(out[6:8], [0.0, 1.0, -0.5])     # defensa: +α al rival por defecto
    assert not out[[0, 1, 4, 5, 8, 9]].any()
    otra = sesgo([_i(6, 8, "DEFENDING IN DEFENSIVE THIRD", H)], 10,
                 params=ParamsSesgo(alfa=1.0, defensa_es_rival=False))
    assert np.allclose(otra[6:8], [1.0, 0.0, 0.0])
    sin_equipo = sesgo([_i(2, 4, "PROGRESSION")], 10, params=p)
    assert not sin_equipo.any()                         # N/A en una fase: nada


def test_sesgo_solapes_no_suman_y_tope():
    inst = [_i(0, 4, "BUILD UP", A), _i(1, 3, "BALL IN FINAL THIRD", A)]
    out = sesgo(inst, 4, params=ParamsSesgo(alfa=1.5))
    assert np.allclose(out[:, A], 1.5)                  # indicador por paso, no suma
    grande = sesgo(inst, 4, params=ParamsSesgo(alfa=9.0, gamma=9.0))
    assert np.allclose(grande[:, A], 2.0) and np.allclose(grande[:, D], -2.0)
    # α y β sobre la misma celda: la suma también se recorta a 2 nats
    mezcla = sesgo([_i(0, 4, "BUILD UP", H), _i(0, 1, "SET PIECES", H, "Corner Kick")], 8,
                   params=ParamsSesgo(alfa=1.5, beta=1.5, ventana_despues_s=0.4))
    assert mezcla.max() <= 2.0 + 1e-9 and np.isclose(mezcla[1, H], 2.0)


def test_sesgo_saques_ventanas():
    p = ParamsSesgo(beta=1.0, ventana_antes_s=0.4, ventana_despues_s=0.4)   # 2 pasos
    out = sesgo([_i(5, 6, "SET PIECES", A, "Goal Kick")], 12, params=p)
    assert np.allclose(out[3:5], [0.0, 0.0, 1.0])       # D antes del inicio del tag
    assert not out[5].any()                              # dentro del tag, neutro
    assert np.allclose(out[6:8], [0.0, 1.0, 0.0])        # equipo que saca, tras el tag
    assert not out[[0, 1, 2, 8, 9, 10, 11]].any()
    banda = sesgo([_i(5, 6, "SET PIECES", tipo="Throw-in")], 12, params=p)
    assert np.allclose(banda[3:5, D], 1.0) and not banda[:, :2].any()   # N/A: solo D


def test_sesgo_reglas_duras():
    sec = np.zeros(40 * K, np.int8)
    sec[2 * K:18 * K] = 1
    sec[22 * K:] = 2
    inst = [_i(0, 2, "SET PIECES", tipo="Kick Off (Start)", half=1),
            _i(16, 17, "BUILD UP", H),
            _i(26, 27, "SET PIECES", tipo="Kick Off (Start)", half=2),
            _i(30, 32, "GOALS", half=2),
            _i(35, 36, "SET PIECES", A, "Kick Off (After Goal)", half=2)]
    p = ParamsSesgo(reglas_duras=True)
    out = sesgo(inst, 40, params=p, seccion=sec)
    duro = out[:, H] < -10
    assert duro[22:26].all() and not duro[16:22].any()     # desde el inicio de la parte 2
    assert duro[32:35].all() and not duro[[30, 31, 35]].any()
    assert (out[duro][:, H] == out[duro][:, A]).all() and not out[:, D].any()
    assert not duro[:22].any()
    # sin `seccion`, la ventana se corta en el fin de la instancia anterior
    sin_sec = sesgo(inst, 40, params=p)
    assert (sin_sec[17:26, H] < -10).all() and not (sin_sec[:17, H] < -10).any()


# ---------- simulador


def _skillcorner_sintetico(raiz, mid="7"):
    d = raiz / "skillcorner" / "matches" / mid
    d.mkdir(parents=True)
    meta = {"home_team": {"id": 1}, "away_team": {"id": 2},
            "match_periods": [{"period": 1, "start_frame": 10, "end_frame": 3009},
                              {"period": 2, "start_frame": 3100, "end_frame": 6099}]}
    (d / "{}_match.json".format(mid)).write_text(json.dumps(meta))
    tipos = [("build_up", "high_block", "middle_third", "False"),
             ("create", "medium_block", "attacking_third", "False"),
             ("finish", "low_block", "attacking_third", "True"),
             ("quick_break", "defending_quick_break", "attacking_third", "False"),
             ("chaotic", "chaotic", "middle_third", "False")]
    with open(d / "{}_phases_of_play.csv".format(mid), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_start", "frame_end", "period", "team_in_possession_id",
                    "team_in_possession_phase_type", "team_out_of_possession_phase_type",
                    "third_end", "penalty_area_end"])
        for k in range(60):
            a = 20 + k * 98 + (190 if k >= 30 else 0)
            t = tipos[k % len(tipos)]
            w.writerow([a, a + 60, 1 if k < 30 else 2, 1 + k % 2, t[0], t[1], t[2], t[3]])
    reanuda = ["throw_in_for", "free_kick_against", "goal_kick_for", "corner_for", ""]
    with open(d / "{}_dynamic_events.csv".format(mid), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "frame_start", "frame_end", "period", "team_id",
                    "game_interruption_before", "game_interruption_after"])
        for k in range(40):
            a = 30 + k * 140
            w.writerow(["player_possession", a, a + 20, 1 if a < 3009 else 2, 1 + k % 2,
                        reanuda[k % len(reanuda)],
                        "goal_for" if k == 12 else "goal_kick_against" if k == 30 else ""])
        w.writerow(["player_possession", 2000, 2010, 1, 2, "goal_against", ""])
        w.writerow(["passing_option", 50, 60, 1, 1, "throw_in_for", ""])


def test_simulador_determinista_y_ruido(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_EXTERNAL_DIR", str(tmp_path))
    _skillcorner_sintetico(tmp_path)
    a = simular_tags.simular("skillcorner", "7", 3)
    assert a == simular_tags.simular("skillcorner", "7", 3)
    assert a != simular_tags.simular("skillcorner", "7", 4)
    assert all(isinstance(i, Instancia) and 0 <= i.inicio_frame < i.fin_frame for i in a)

    limpio = simular_tags.simular("skillcorner", "7", 0, simular_tags.SIN_RUIDO)
    assert limpio == simular_tags.simular("skillcorner", "7", 99, simular_tags.SIN_RUIDO)
    codigos = {i.code for i in limpio}
    assert {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
            "BALL IN FINAL THIRD", "BALL IN THE BOX", "DEFENDING IN ATTACKING THIRD",
            "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
            "SET PIECES", "GOALS"} <= codigos
    tipos = {i.type for i in limpio if i.code == "SET PIECES"}
    assert {"Throw-in", "Free Kick", "Goal Kick", "Corner Kick", "Kick Off (Start)",
            "Kick Off (After Goal)"} <= tipos
    assert all(i.team == SIN_EQUIPO for i in limpio if i.type == "Throw-in")
    # el ruido por defecto descarta: menos instancias que sin ruido
    assert len(a) < len(limpio)
    # la segunda parte empieza tras la primera más 25 frames de separación
    assert sum(i.code == "GOALS" for i in limpio) == 1           # goal_kick_* no es gol
    assert sum(i.type == "Kick Off (After Goal)" for i in limpio) == 1
    ko = sorted(i.inicio_frame for i in limpio if i.type == "Kick Off (Start)")
    assert ko == [0, (3009 - 10) * 25 // 10 + 1 + 25]


def test_simulador_fuente_sin_eventos():
    with pytest.raises(NotImplementedError):
        simular_tags.simular("idsse", "J03WMX", 0)
