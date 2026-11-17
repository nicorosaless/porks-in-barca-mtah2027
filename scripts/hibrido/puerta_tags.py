#!/usr/bin/env python3
"""Puerta de E4 (docs/PLAN-IT9.md): ¿el equipo del Smart Tagging concuerda con el que predice it8?

Comprobación automática SIN etiquetas: solo usa el ATD, el Smart Tagging y las predicciones del
modelo vigente sobre BAR-ATH. Ni GT ni vídeo. Imprime solo agregados (fracciones por grupo de
códigos). Umbral fijado antes de mirar: concordancia >= 80 % en pasos con balón visible y
max P(H, A) >= 0.9; si no, E4 solo usa las reglas duras.

    python scripts/hibrido/puerta_tags.py [--modelo it8]
"""
import argparse
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from mtah.fuentes import reto, smart_tagging  # noqa: E402
from mtah.hibrido import rasgos as R  # noqa: E402
from mtah.hibrido.decodificar import PASO  # noqa: E402
from mtah.hibrido.sesgo_tags import ATAQUE, DEFENSA  # noqa: E402
from mtah.predictores.hibrido import MODELOS, _logp_red, mezclar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="it8")
    ap.add_argument("--umbral_p", type=float, default=0.9)
    args = ap.parse_args()
    p = reto.cargar(gt_publico=False)
    cfg = json.load(open(os.path.join(MODELOS, args.modelo + ".json")))
    r = R.rasgos(p, gt=None)
    comp = cfg.get("componentes") or [{"pesos_red": cfg["pesos_red"], "peso": 1.0,
                                        "convencion": cfg.get("convencion")}]
    logp = mezclar([_logp_red(r, c["pesos_red"], c.get("convencion")) for c in comp],
                   [c["peso"] for c in comp])
    prob = np.exp(logp)
    s_n = len(prob)
    visible = np.zeros(s_n, bool)
    vis_f = ~np.isnan(p.balon[:, 0])
    for s in range(s_n):
        visible[s] = vis_f[s * PASO:(s + 1) * PASO].mean() >= 0.6
    vivo_seguro = (prob[:, :2].max(1) >= args.umbral_p)
    pred_eq = prob[:, :2].argmax(1)  # 0 = H, 1 = A
    inst = smart_tagging.cargar()
    grupos = {"ataque": ATAQUE, "defensa": DEFENSA}
    print("pasos {}  visibles {:.0%}  seguros {:.0%}".format(s_n, visible.mean(), vivo_seguro.mean()))
    for nombre, codigos in grupos.items():
        acuerdo = total = 0
        for i in inst:
            if i.code not in codigos or i.team not in (0, 1):
                continue
            a, b = i.inicio_frame // PASO, min(s_n, (i.fin_frame - 1) // PASO + 1)
            m = visible[a:b] & vivo_seguro[a:b]
            total += int(m.sum())
            acuerdo += int((pred_eq[a:b][m] == i.team).sum())
        print("{:<8} pasos evaluables {:>6}  concordancia Team == equipo predicho {:.1%}".format(
            nombre, total, acuerdo / max(total, 1)))


if __name__ == "__main__":
    main()
