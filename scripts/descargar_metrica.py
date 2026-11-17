#!/usr/bin/env python3
"""Descarga Metrica Sports sample-data desde GitHub.

Metrica Sports publica `metrica-sports/sample-data` como datos de ejemplo abiertos en GitHub. El
README no incluye una licencia formal; en su sección "Legal stuff" pide usar los datos de forma
responsable y reconocer la fuente en cualquier uso público. En este repo se usa solo como dato
externo de otros partidos para estimar la convención de posesión de Metrica.

Destino: $MTAH_EXTERNAL_DIR/metrica, fuera del repo y de git.
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.rutas import external_dir  # noqa: E402

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
FICHEROS = {
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv": 139446,
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073,
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv": 153897,
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": 31890338,
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": 29627711,
    "data/Sample_Game_3/Sample_Game_3_events.json": 1709486,
    "data/Sample_Game_3/Sample_Game_3_metadata.xml": 81806,
    "data/Sample_Game_3/Sample_Game_3_tracking.txt": 53443115,
}


def descargar(rel, tamano, destino):
    local = rel.replace("data/", "")
    ruta = os.path.join(destino, local)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    hecho = os.path.getsize(ruta) if os.path.exists(ruta) else 0
    if hecho == tamano:
        print("ok      {}".format(local))
        return
    if hecho > tamano:
        os.remove(ruta)
        hecho = 0
    req = urllib.request.Request(BASE + rel)
    if hecho:
        req.add_header("Range", "bytes={}-".format(hecho))
    print("bajando {} ({:.1f} MB, desde {:.1f} MB)".format(local, tamano / 1e6, hecho / 1e6),
          flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp, open(ruta, "ab" if hecho else "wb") as f:
        if hecho and resp.status != 206:
            raise RuntimeError("el servidor no aceptó reanudar {}".format(local))
        while True:
            bloque = resp.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
    final = os.path.getsize(ruta)
    if final != tamano:
        raise RuntimeError("{}: {} bytes, se esperaban {}".format(local, final, tamano))


def main():
    destino = os.path.join(external_dir(), "metrica")
    for rel, tamano in FICHEROS.items():
        descargar(rel, tamano, destino)
    print("listo: {}".format(destino))


if __name__ == "__main__":
    main()
