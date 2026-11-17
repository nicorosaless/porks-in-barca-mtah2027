"""Convierte un partido IDSSE (DFL/TRACAB, 25 fps) en un `Partido` con GT por frame.

El GT se deriva igual que el del reto: tracking corregido, y por frame
    BallStatus == 0                 -> D
    BallStatus == 1, posesión local -> H
    BallStatus == 1, posesión vis.  -> A
Si falta BallPossession en un frame vivo, el frame queda X (no se puntúa). kloppy, en cambio, lo
asigna en silencio al visitante; por eso este parser es propio.

Datos: Bassek, Rein, Weber, Memmert (2025), Scientific Data 12, 195. CC BY 4.0.
Coordenadas DFL: metros, origen en el centro, Y hacia arriba. Se pasan a 0-1 con y hacia abajo,
la convención del ATD de Metrica.
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from mtah.partido import A, D, H, X, Partido
from mtah.rutas import cache_dir, external_dir

PARTIDOS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECCIONES = {"firstHalf": 1, "secondHalf": 2}
SEPARACION = 25  # frames X entre partes, para que el scorer no cruce de una a otra


def _fichero(match_id, prefijo):
    patron = os.path.join(external_dir(), "idsse", "{}*-{}.xml".format(prefijo, match_id))
    encontrados = glob.glob(patron)
    if len(encontrados) != 1:
        raise FileNotFoundError("{}: ejecuta scripts/descargar_idsse.py {}".format(patron, match_id))
    return encontrados[0]


def _info(match_id):
    root = ET.parse(_fichero(match_id, "DFL_02_01_matchinformation_")).getroot()
    env = root.find(".//Environment")
    campo = (float(env.get("PitchX")), float(env.get("PitchY")))
    rol = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}
    local = [tid for tid, r in rol.items() if r == "home"]
    assert len(local) == 1, rol
    general = root.find(".//General")
    return campo, local[0], general.get("MatchTitle")


def _leer_posiciones(match_id):
    """{seccion: {"ball": {N: (x, y, posesion, estado)}, persona: (equipo, {N: (x, y)})}}"""
    datos = {1: {}, 2: {}}
    ruta = _fichero(match_id, "DFL_04_03_positions_raw_observed_")
    actual = None
    for evento, el in ET.iterparse(ruta, events=("start", "end")):
        if evento == "start" and el.tag == "FrameSet":
            sec = SECCIONES.get(el.get("GameSection"))
            equipo, persona = el.get("TeamId"), el.get("PersonId")
            if sec is None or equipo == "referee":
                actual = None
            elif equipo == "BALL":
                actual = datos[sec].setdefault("ball", {})
            else:
                actual = datos[sec].setdefault(persona, (equipo, {}))[1]
            es_balon = equipo == "BALL"
        elif evento == "end" and el.tag == "Frame":
            if actual is not None:
                n = int(el.get("N"))
                x, y = float(el.get("X")), float(el.get("Y"))
                if es_balon:
                    pos = el.get("BallPossession")
                    actual[n] = (x, y, int(pos) if pos else 0, int(el.get("BallStatus", "0")))
                else:
                    actual[n] = (x, y)
            el.clear()
        elif evento == "end" and el.tag == "FrameSet":
            el.clear()
    return datos


def convertir(match_id):
    campo, local, titulo = _info(match_id)
    largo, ancho = campo
    datos = _leer_posiciones(match_id)

    personas = sorted({p for sec in datos.values() for p in sec if p != "ball"})
    equipo_de = {}
    for sec in datos.values():
        for p, v in sec.items():
            if p != "ball":
                equipo_de[p] = 0 if v[0] == local else 1
    slot = {p: k for k, p in enumerate(personas)}
    k_total = len(personas)

    bloques_balon, bloques_jug, bloques_gt, bloques_sec = [], [], [], []
    for sec in (1, 2):
        balon = datos[sec]["ball"]
        ns = np.array(sorted(balon))
        n0, t = ns[0], ns[-1] - ns[0] + 1
        b = np.full((t, 2), np.nan, np.float32)
        gt = np.full(t, X, np.int8)
        for n, (x, y, pos, estado) in balon.items():
            i = n - n0
            b[i] = ((x + largo / 2) / largo, (ancho / 2 - y) / ancho)
            if estado == 0:
                gt[i] = D
            elif pos == 1:
                gt[i] = H
            elif pos == 2:
                gt[i] = A
        j = np.full((t, k_total, 2), np.nan, np.float32)
        for p, (_, frames) in ((p, v) for p, v in datos[sec].items() if p != "ball"):
            idx = np.fromiter(frames.keys(), np.int64) - n0
            xy = np.array(list(frames.values()), np.float32)
            ok = (idx >= 0) & (idx < t)
            j[idx[ok], slot[p], 0] = (xy[ok, 0] + largo / 2) / largo
            j[idx[ok], slot[p], 1] = (ancho / 2 - xy[ok, 1]) / ancho
        bloques_balon += [b, np.full((SEPARACION, 2), np.nan, np.float32)]
        bloques_jug += [j, np.full((SEPARACION, k_total, 2), np.nan, np.float32)]
        bloques_gt += [gt, np.full(SEPARACION, X, np.int8)]
        bloques_sec += [np.full(t, sec, np.int8), np.zeros(SEPARACION, np.int8)]

    p = Partido(nombre="idsse_{}".format(match_id), campo=campo,
                balon=np.concatenate(bloques_balon), jugadores=np.concatenate(bloques_jug),
                equipo_slot=np.array([equipo_de[q] for q in personas], np.int8),
                seccion=np.concatenate(bloques_sec), gt=np.concatenate(bloques_gt))
    p.extra["titulo"] = titulo
    return p.validar()


def cargar(match_id, usar_cache=True):
    os.makedirs(cache_dir(), exist_ok=True)
    ruta = os.path.join(cache_dir(), "idsse_{}.npz".format(match_id))
    if usar_cache and os.path.exists(ruta):
        return Partido.cargar(ruta)
    p = convertir(match_id)
    p.guardar(ruta)
    return p


def disponibles():
    # vale el XML original o su conversión en cache (en MareNostrum solo se sube la cache)
    return [m for m in PARTIDOS
            if glob.glob(os.path.join(external_dir(), "idsse", "DFL_04_03_*-{}.xml".format(m)))
            or os.path.exists(os.path.join(cache_dir(), "idsse_{}.npz".format(m)))]
