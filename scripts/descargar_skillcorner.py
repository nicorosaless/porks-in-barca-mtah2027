#!/usr/bin/env python3
"""Descarga SkillCorner Open Data.

Fuente: https://github.com/SkillCorner/opendata, rama master, licencia MIT. Los ficheros de
tracking están en Git LFS; si no hay `git-lfs`, se descargan desde media.githubusercontent.com.

Destino: $MTAH_EXTERNAL_DIR/skillcorner, fuera del repo y de git.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.rutas import external_dir  # noqa: E402

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/"
FICHEROS_PARTIDO = (
    "{id}_match.json",
    "{id}_dynamic_events.csv",
    "{id}_phases_of_play.csv",
    "{id}_tracking_extrapolated.jsonl",
)


def _descargar(url, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    hecho = os.path.getsize(ruta) if os.path.exists(ruta) else 0
    try:
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as resp:
            tamano = int(resp.headers.get("Content-Length", "0") or 0)
        if tamano and hecho == tamano:
            print("ok      {}".format(os.path.basename(ruta)))
            return
        if tamano and hecho > tamano:
            os.remove(ruta)
            hecho = 0
    except Exception:
        pass
    req = urllib.request.Request(url)
    if hecho:
        req.add_header("Range", "bytes={}-".format(hecho))
    modo = "ab" if hecho else "wb"
    print("bajando {} desde {:.1f} MB".format(os.path.basename(ruta), hecho / 1e6), flush=True)
    with urllib.request.urlopen(req, timeout=180) as resp, open(ruta, modo) as f:
        if hecho and resp.status != 206:
            f.seek(0)
            f.truncate()
        while True:
            bloque = resp.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)


def _url(rel):
    if rel.endswith("_tracking_extrapolated.jsonl"):
        return MEDIA + rel
    return RAW + rel


def main():
    destino = os.path.join(external_dir(), "skillcorner")
    matches_rel = "data/matches.json"
    matches_local = os.path.join(destino, "matches.json")
    _descargar(RAW + matches_rel, matches_local)
    matches = json.load(open(matches_local))
    if isinstance(matches, dict):
        matches = matches.get("matches", matches.get("data", []))
    ids = [str(m["id"] if isinstance(m, dict) else m) for m in matches]
    for mid in ids:
        for plantilla in FICHEROS_PARTIDO:
            nombre = plantilla.format(id=mid)
            rel = "data/matches/{}/{}".format(mid, nombre)
            local = os.path.join(destino, "matches", mid, nombre)
            _descargar(_url(rel), local)
    print("listo: {} partidos en {}".format(len(ids), destino))


if __name__ == "__main__":
    main()
