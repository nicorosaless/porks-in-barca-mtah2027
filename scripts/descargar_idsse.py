#!/usr/bin/env python3
"""Descarga el dataset abierto IDSSE (DFL, 7 partidos de Bundesliga 2022/23) desde Hugging Face.

Es el conjunto de validación externo del equipo: tracking a 25 fps con, en cada frame, el flag
de balón vivo/muerto (BallStatus) y el equipo en posesión (BallPossession). Los organizadores
confirmaron en la issue #16 del repo oficial que se puede usar (otros partidos, CC BY 4.0).

Cita obligatoria (CC BY 4.0): Bassek, M., Rein, R., Weber, H., Memmert, D. "An integrated dataset
of spatiotemporal and event data in elite soccer." Scientific Data 12, 195 (2025).
https://doi.org/10.1038/s41597-025-04505-y

Uso:
    python scripts/descargar_idsse.py                 # los 7 partidos, ~2,6 GB
    python scripts/descargar_idsse.py J03WPY J03WN1   # solo algunos

Destino: $MTAH_EXTERNAL_DIR/idsse (por defecto data/external/idsse, ignorado por git).
Reanuda descargas a medias y se salta los ficheros que ya están completos.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.rutas import external_dir  # noqa: E402

REPO = "pysport/idsse-data"
API = "https://huggingface.co/api/datasets/{}/tree/main".format(REPO)
FILE_URL = "https://huggingface.co/datasets/{}/resolve/main/{{}}".format(REPO)

PARTIDOS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
# Solo hace falta la información del partido (equipos, dimensiones) y las posiciones.
# Los eventos no se descargan: el GT sale de los flags del tracking.
PREFIJOS = ("DFL_02_01_matchinformation_", "DFL_04_03_positions_raw_observed_")


def listar():
    with urllib.request.urlopen(API, timeout=60) as resp:
        return {f["path"]: f["size"] for f in json.load(resp) if f.get("type") == "file"}


def descargar(nombre, tamano, destino):
    ruta = os.path.join(destino, nombre)
    hecho = os.path.getsize(ruta) if os.path.exists(ruta) else 0
    if hecho == tamano:
        print("ok      {}".format(nombre))
        return
    if hecho > tamano:
        os.remove(ruta)
        hecho = 0
    req = urllib.request.Request(FILE_URL.format(nombre))
    if hecho:
        req.add_header("Range", "bytes={}-".format(hecho))
    print("bajando {} ({:.0f} MB, desde {:.0f} MB)".format(nombre, tamano / 1e6, hecho / 1e6),
          flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp, open(ruta, "ab" if hecho else "wb") as f:
        if hecho and resp.status != 206:
            raise RuntimeError("el servidor no aceptó reanudar {}".format(nombre))
        while True:
            bloque = resp.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
    final = os.path.getsize(ruta)
    if final != tamano:
        raise RuntimeError("{}: {} bytes, se esperaban {}".format(nombre, final, tamano))


def main(argv):
    pedidos = argv or list(PARTIDOS)
    desconocidos = set(pedidos) - set(PARTIDOS)
    if desconocidos:
        sys.exit("partidos desconocidos: {}".format(sorted(desconocidos)))
    destino = os.path.join(external_dir(), "idsse")
    os.makedirs(destino, exist_ok=True)
    ficheros = listar()
    for nombre, tamano in sorted(ficheros.items()):
        if nombre.startswith(PREFIJOS) and any(nombre.endswith("-{}.xml".format(p)) for p in pedidos):
            descargar(nombre, tamano, destino)
    print("listo: {}".format(destino))


if __name__ == "__main__":
    main(sys.argv[1:])
