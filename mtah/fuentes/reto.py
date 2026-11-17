"""Carga el ATD táctico del reto en un `Partido`.

Lee los mismos ficheros que starter/baseline.py, con el mismo criterio de equipos (el primer
equipo del XML es el local). El resultado se cachea en data/cache, que git ignora: es un derivado
del dataset y no se publica nunca (RULES.md, "Do not publish derived data").
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from mtah.partido import Partido, X
from mtah.rutas import RAIZ

sys.path.insert(0, RAIZ)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE = os.path.join(RAIZ, "data", "cache")


def _equipos(meta_path):
    root = ET.parse(meta_path).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    if len(ids) != 2:
        raise ValueError("{}: se esperaban 2 equipos".format(meta_path))
    return np.array([0 if p.get("teamId") == ids[0] else 1 for p in root.iter("Player")], np.int8)


def _parsear(raw_path, n_frames, n_slots):
    balon = np.full((n_frames, 2), np.nan, np.float32)
    jug = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    with open(raw_path) as f:
        for linea in f:
            partes = linea.rstrip("\n").split(":")
            if len(partes) != 3:
                continue
            t = int(partes[0]) - 1          # frame_id empieza en 1; las filas de entrega en 0
            if not 0 <= t < n_frames:
                continue
            b = partes[2].split(",")
            if b[0][:1] != "N":
                balon[t] = (float(b[0]), float(b[1]))
            for k, g in enumerate(partes[1].split(";")[:n_slots]):
                if g[:1] != "N":
                    c = g.split(",")
                    jug[t, k] = (float(c[0]), float(c[1]))
    return balon, jug


def cargar(gt_publico=True, game=GAME, usar_cache=True):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "{}_tactical.npz".format(game))
    if usar_cache and os.path.exists(cache):
        p = Partido.cargar(cache)
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
        equipo_slot = _equipos(meta_path)
        n = source_frames(game=game)
        balon, jug = _parsear(raw_path, n, len(equipo_slot))
        seccion = np.zeros(n, np.int8)
        for mitad, ini, fin in half_video_blocks(game):
            seccion[ini:fin + 1] = mitad
        g = GAMES[game]
        p = Partido(nombre="{}_tactical".format(game), campo=(g["pitch"][0], g["pitch"][1]),
                    balon=balon, jugadores=jug, equipo_slot=equipo_slot, seccion=seccion).validar()
        p.guardar(cache)
    if gt_publico:
        p.gt = _gt_publico(game, p.n_frames)
    return p


def _gt_publico(game, n):
    """GT público: X fuera de los 5 minutos. Solo para comprobar alineación, nunca para ajustar."""
    sys.path.insert(0, os.path.join(RAIZ, "evaluation"))
    import possession_metrics as pm
    ruta = os.path.join(RAIZ, "evaluation", "gt", "{}_possession_gt_public.csv".format(game))
    codigos = pm.label_codes(pm.read_label_csv(ruta)).astype(np.int8)
    assert len(codigos) == n and (codigos != X).any()
    return codigos
