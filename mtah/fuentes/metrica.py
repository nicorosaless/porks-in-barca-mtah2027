"""Metrica Sports sample-data -> `Partido`.

Los tres partidos son públicos en https://github.com/metrica-sports/sample-data. El tracking usa
la convención Metrica normalizada 0-1, con y hacia abajo, y las etiquetas H/A/D se derivan de los
eventos publicados.
"""
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

from mtah.partido import A, D, H, X, Partido
from mtah.rutas import cache_dir, external_dir

PARTIDOS = ("SG1", "SG2", "SG3")
SEPARACION = 25


@dataclass(frozen=True)
class Evento:
    equipo: int
    tipo: str
    subtipos: str
    periodo: int
    inicio: int
    fin: int


def _dir():
    return os.path.join(external_dir(), "metrica")


def _ruta(sg, nombre):
    return os.path.join(_dir(), "Sample_Game_{}".format(sg), nombre)


def _subtipos(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_subtipos(x) for x in v)
    if isinstance(v, dict):
        return str(v.get("name", ""))
    return str(v)


def _es_tiro_fuera_o_gol(tipo, subtipos):
    tipo = tipo.upper()
    sub = subtipos.upper()
    return tipo == "SHOT" and (("GOAL" in sub and "WOODWORK" not in sub) or "OUT" in sub)


def _es_parada(ev):
    tipo = ev.tipo.upper()
    sub = ev.subtipos.upper()
    if tipo in {"BALL OUT", "FAULT RECEIVED", "CARD"}:
        return True
    if tipo == "CHALLENGE" and "FAULT" in sub:
        return True
    if tipo == "BALL LOST" and any(x in sub for x in ("END HALF", "OFFSIDE", "HAND BALL", "GOAL KICK")):
        return True
    return _es_tiro_fuera_o_gol(tipo, sub)


def etiquetas_desde_eventos(eventos, n_frames, seccion=None, reglas_sg3=False):
    """Eventos normalizados -> GT por frame.

    Cada parte empieza en D. `BALL LOST` no cambia la posesión; el primer evento de control del
    rival la cambia. `BALL OUT`, faltas y goles pasan a D según la convención del plan.
    """
    gt = np.full(n_frames, D, np.int8)
    if seccion is None:
        seccion = np.ones(n_frames, np.int8)
    vivos_desde_d = 0
    ordenados = sorted(eventos, key=lambda e: (e.inicio, e.periodo))
    for ev in ordenados:
        ini = max(0, min(int(ev.inicio), n_frames - 1))
        fin = max(0, min(int(ev.fin), n_frames - 1))
        tipo = ev.tipo.upper()
        sub = ev.subtipos.upper()
        equipo = H if ev.equipo == 0 else A
        if tipo == "SET PIECE":
            gt[ini:][seccion[ini:] == seccion[ini]] = equipo
        elif tipo in {"PASS", "CARRY", "SHOT", "RECOVERY"}:
            if gt[ini] == D:
                vivos_desde_d += 1
            gt[ini:][seccion[ini:] == seccion[ini]] = equipo
            if (tipo == "SHOT" and "GOAL" in sub and "WOODWORK" not in sub) or (
                reglas_sg3 and _es_tiro_fuera_o_gol(tipo, sub)
            ):
                gt[fin:][seccion[fin:] == seccion[fin]] = D
        elif tipo == "BALL LOST":
            if "END HALF" in sub or (
                reglas_sg3 and any(x in sub for x in ("OFFSIDE", "HAND BALL", "GOAL KICK"))
            ):
                gt[ini:][seccion[ini:] == seccion[ini]] = D
        elif tipo == "BALL OUT":
            desde = fin if fin > ini else ini
            gt[desde:][seccion[desde:] == seccion[desde]] = D
        elif tipo == "FAULT RECEIVED" or (
            reglas_sg3 and ((tipo == "CHALLENGE" and "FAULT" in sub) or tipo == "CARD")
        ):
            gt[ini:][seccion[ini:] == seccion[ini]] = D
        elif tipo == "CHALLENGE":
            pass
    for prev, ev in zip(ordenados, ordenados[1:]):
        if reglas_sg3 and ev.tipo.upper() == "SET PIECE" and prev.tipo.upper() in {
            "BALL LOST", "CARRY", "CHALLENGE",
        }:
            ini = max(0, min(int(prev.fin), n_frames))
            fin = max(0, min(int(ev.inicio), n_frames))
            if fin > ini:
                gt[ini:fin] = D
    if vivos_desde_d:
        print("eventos vivos desde D sin SET PIECE anotado: {}".format(vivos_desde_d))
    gt[seccion == 0] = X
    return gt


def _eventos_csv(ruta):
    eventos = []
    with open(ruta, newline="") as f:
        for row in csv.DictReader(f):
            equipo = 0 if row["Team"] == "Home" else 1
            ini = int(float(row["Start Frame"] or 0))
            fin = int(float(row["End Frame"] or ini))
            if fin <= 0:
                fin = ini
            eventos.append(Evento(equipo, row["Type"], row.get("Subtype", ""), int(row["Period"]),
                                  ini, fin))
    return eventos


def _eventos_json(ruta):
    data = json.load(open(ruta))["data"]
    equipos = sorted({e["team"]["id"] for e in data if e.get("team")})
    mapa = {equipos[0]: 0, equipos[1]: 1}
    eventos = []
    for e in data:
        if not e.get("team"):
            continue
        eventos.append(Evento(
            mapa[e["team"]["id"]],
            e["type"]["name"],
            _subtipos(e.get("subtypes")),
            int(e["period"]),
            int(e["start"]["frame"]),
            int(e.get("end", e["start"])["frame"]),
        ))
    return eventos


def _leer_tracking_csv(home, away):
    import pandas as pd

    def leer(ruta):
        with open(ruta, newline="") as f:
            cab = [next(csv.reader(f)) for _ in range(3)]
        df = pd.read_csv(ruta, skiprows=3, header=None)
        jugadores = []
        ball_i = None
        for i in range(3, len(cab[2]) - 1, 2):
            nombre = cab[2][i].strip()
            if nombre == "Ball":
                ball_i = i
                break
            if nombre:
                jugadores.append((nombre, i, i + 1))
        if ball_i is None:
            ball_i = len(cab[2]) - 2
        return df, jugadores, (ball_i, ball_i + 1)

    dh, jugadores_h, ball_cols = leer(home)
    da, jugadores_a, _ = leer(away)
    bloques = []
    equipos = []
    for periodo in (1, 2):
        h = dh[dh[0] == periodo].reset_index(drop=True)
        a = da[da[0] == periodo].reset_index(drop=True)
        frames = h[1].to_numpy(np.int64)
        n = len(frames)
        balon = h[list(ball_cols)].to_numpy(np.float32)
        jug = np.full((n, len(jugadores_h) + len(jugadores_a), 2), np.nan, np.float32)
        for k, (_, x, y) in enumerate(jugadores_h):
            jug[:, k] = h[[x, y]].to_numpy(np.float32)
        for j, (_, x, y) in enumerate(jugadores_a):
            jug[:, len(jugadores_h) + j] = a[[x, y]].to_numpy(np.float32)
        bloques.append((periodo, frames, balon, jug))
        equipos = [0] * len(jugadores_h) + [1] * len(jugadores_a)
    return bloques, np.array(equipos, np.int8)


def _leer_tracking_sg3(meta, raw):
    from kloppy import metrica

    ds = metrica.load_tracking_epts(meta_data=meta, raw_data=raw, coordinates="metrica")
    df = ds.to_df()
    jugadores = []
    equipo = []
    for eq, team in enumerate(ds.metadata.teams):
        for p in team.players:
            jugadores.append(str(p.player_id))
            equipo.append(eq)
    bloques = []
    for periodo, lo, hi in ((1, 1, 69661), (2, 69662, 143761)):
        d = df[(df["frame_id"] >= lo) & (df["frame_id"] <= hi)].reset_index(drop=True)
        frames = d["frame_id"].to_numpy(np.int64)
        balon = d[["ball_x", "ball_y"]].to_numpy(np.float32)
        jug = np.full((len(d), len(jugadores), 2), np.nan, np.float32)
        for k, pid in enumerate(jugadores):
            x, y = pid + "_x", pid + "_y"
            if x in d.columns and y in d.columns:
                jug[:, k] = d[[x, y]].to_numpy(np.float32)
        bloques.append((periodo, frames, balon, jug))
    return bloques, np.array(equipo, np.int8)


def _convertir(sg):
    if sg in (1, 2):
        bloques, equipo_slot = _leer_tracking_csv(
            _ruta(sg, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(sg)),
            _ruta(sg, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(sg)),
        )
        eventos = _eventos_csv(_ruta(sg, "Sample_Game_{}_RawEventsData.csv".format(sg)))
    else:
        bloques, equipo_slot = _leer_tracking_sg3(
            _ruta(3, "Sample_Game_3_metadata.xml"),
            _ruta(3, "Sample_Game_3_tracking.txt"),
        )
        eventos = _eventos_json(_ruta(3, "Sample_Game_3_events.json"))

    out_b, out_j, out_sec, out_gt = [], [], [], []
    for periodo, frames, balon, jug in bloques:
        n = len(frames)
        sec = np.full(n, periodo, np.int8)
        evs = []
        primero = int(frames[0])
        for e in eventos:
            if e.periodo == periodo:
                evs.append(Evento(e.equipo, e.tipo, e.subtipos, periodo,
                                  int(e.inicio - primero), int(e.fin - primero)))
        gt = etiquetas_desde_eventos(evs, n, sec, reglas_sg3=(sg == 3))
        out_b += [balon, np.full((SEPARACION, 2), np.nan, np.float32)]
        out_j += [jug, np.full((SEPARACION, jug.shape[1], 2), np.nan, np.float32)]
        out_sec += [sec, np.zeros(SEPARACION, np.int8)]
        out_gt += [gt, np.full(SEPARACION, X, np.int8)]
    p = Partido("metrica_SG{}".format(sg), (105.0, 68.0), np.concatenate(out_b),
                np.concatenate(out_j), equipo_slot, np.concatenate(out_sec),
                np.concatenate(out_gt)).validar()
    return p


def cargar(id, usar_cache=True):
    if id not in PARTIDOS:
        raise ValueError("partido Metrica desconocido: {}".format(id))
    os.makedirs(cache_dir(), exist_ok=True)
    ruta = os.path.join(cache_dir(), "metrica_{}.npz".format(id))
    if usar_cache and os.path.exists(ruta):
        return Partido.cargar(ruta)
    p = _convertir(int(id[-1]))
    p.guardar(ruta)
    return p


def disponibles():
    out = []
    for p in PARTIDOS:
        if os.path.exists(os.path.join(cache_dir(), "metrica_{}.npz".format(p))):
            out.append(p)
            continue
        sg = int(p[-1])
        if sg in (1, 2):
            ok = os.path.exists(_ruta(sg, "Sample_Game_{}_RawEventsData.csv".format(sg)))
        else:
            ok = os.path.exists(_ruta(3, "Sample_Game_3_events.json"))
        if ok:
            out.append(p)
    return out
