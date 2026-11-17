#!/usr/bin/env python3
"""Aplica la regla prerregistrada de E5 (docs/HIBRIDO.md) a los JSON de afinar.py.

v4 pasa si Δ ZX-dev >= max(0,02, 2σ) y Δ SI-DFL >= −0,02. σ es la mayor desviación típica entre
semillas (ddof 1) de los dos brazos.

    python scripts/hibrido/decidir_e5.py [--dir salidas/hibrido]
"""
import argparse
import json
import os

import numpy as np


def cargar(d, etiqueta, metrica):
    return json.load(open(os.path.join(d, "it9_E5_{}_{}.json".format(etiqueta, metrica))))


def decidir(base_zx, v4_zx, base_si, v4_si):
    sigma = max(np.std(base_zx, ddof=1), np.std(v4_zx, ddof=1))
    d_zx = float(np.mean(v4_zx) - np.mean(base_zx))
    d_si = float(np.mean(v4_si) - np.mean(base_si))
    umbral = max(0.02, 2 * sigma)
    return dict(d_zx=d_zx, d_si=d_si, sigma=float(sigma), umbral=umbral,
                pasa=bool(d_zx >= umbral and d_si >= -0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="salidas/hibrido")
    args = ap.parse_args()
    brazos = {"base": ["base0", "base1", "base2"], "v4": ["v4_0", "v4_1", "v4_2"]}
    zx, si, por_partido = {}, {}, {}
    for b, etiquetas in brazos.items():
        zx[b] = [cargar(args.dir, e, "ZXdev")["spell_f1"] for e in etiquetas]
        si[b] = [cargar(args.dir, e, "SI")["spell_f1"] for e in etiquetas]
        pp = [cargar(args.dir, e, "ZXdev")["por_partido"] for e in etiquetas]
        por_partido[b] = {k: np.mean([p[k] for p in pp]) for k in pp[0]}
        print("{:<5} ZX-dev {}  media {:.4f} | SI-DFL {}  media {:.4f}".format(
            b, " ".join("{:.4f}".format(x) for x in zx[b]), np.mean(zx[b]),
            " ".join("{:.4f}".format(x) for x in si[b]), np.mean(si[b])))
    r = decidir(zx["base"], zx["v4"], si["base"], si["v4"])
    ks = sorted(por_partido["base"])
    gana = sum(por_partido["v4"][k] > por_partido["base"][k] for k in ks)
    print("Δ ZX-dev {:+.4f} (umbral {:.4f}, σ {:.4f}; gana {}/{} partidos)  Δ SI-DFL {:+.4f}  -> {}".format(
        r["d_zx"], r["umbral"], r["sigma"], gana, len(ks), r["d_si"], "PASA" if r["pasa"] else "NO PASA"))


if __name__ == "__main__":
    main()
