#!/usr/bin/env python3
"""E0: dónde se pierde spellF1. Clasifica cada posesión GT de los partidos externos dev.

Decodifica los log P fuera de muestra (OOF) de un run con costes fijos y asigna a cada posesión
GT (racha máxima de H o de A dentro de un tramo evaluable) una clase. Las reglas se aplican en
este orden y gana la primera que se cumple:

    ok       emparejada en el emparejamiento 1 a 1 oficial (voraz por IoU) con umbral 0,5.
    borde    no es ok y su mejor IoU con una racha predicha del mismo equipo está en [0,3, 0,5):
             la posesión se ve pero los bordes están corridos. Se compara con el mejor IoU, no
             con el emparejamiento, para no confundir un borde con una racha robada.
    equipo   más del 50 % de sus frames se predicen del otro equipo.
    perdida  más del 50 % de sus frames se predicen D.
    fundida  la racha predicha del mismo equipo que más la solapa cubre al menos el 50 % de
             OTRA posesión GT, de cualquier equipo: el modelo se salta un cambio o un parón y
             une dos posesiones. El 50 % hace que rozar el borde de la vecina no cuente. Aquí
             caen también las de IoU ≥ 0,5 que pierden el emparejamiento 1 a 1.
    partida  dos o más rachas predichas del mismo equipo la cubren, cada una con al menos el
             10 % de sus frames, y ninguna con IoU ≥ 0,3 (eso ya sería borde).
    otro     el resto: cobertura mezclada sin mayoría, o una sola racha corta del mismo equipo.

La mayoría de equipo o D va antes que fundida y partida porque es un fallo de emisión, no de
segmentación: si casi toda la posesión sale del otro equipo, da igual cómo se corte.

Cubos: duración GT <3 s, 3–8 s, 8–20 s, >20 s (25 fps). Balón visible: media de balon[:, 0]
(rasgo `vis` de la caché degradada, a 5 fps) sobre los frames de la posesión, en cubos
<0,5, 0,5–0,8, 0,8–0,95 y ≥0,95.

Oráculos (log P cambiado antes de Viterbi, con los mismos costes). El GT por paso es el voto de
sus 5 frames evaluables; un paso es vivo si tiene más frames H+A que D:
    base     sin tocar.
    vivo     pasos muertos: H y A a -inf; pasos vivos: D a -inf.
    equipo   pasos vivos: el otro equipo a -inf; D queda libre.
    ambos    vivo + equipo (techo por la cuantización a 5 fps).

Solo lee: OOF, cachés de rasgos (gt, seccion_frames, balon) y json de costes. Excluye siempre la
caja fuerte. Solo imprime agregados.

    PYTHONPATH=<T-2611-base> MTAH_EXTERNAL_DIR=... python scripts/hibrido/diagnostico.py \\
        <T-2611-base>/salidas/hibrido/runs/it8 --costes <T-2611-base>/modelos/hibrido/it8.json \\
        --fuente_datos idsse --convencion metrica --escenarios v2,pes,v3,v3pes
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

import mtah

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mtah.__file__))),
                                "scripts", "hibrido"))
import afinar as AF  # noqa: E402
from mtah.convencion import MINIMO_S_DEFECTO  # noqa: E402
from mtah.escenarios import ESCENARIOS  # noqa: E402
from mtah.evaluar import pm  # noqa: E402
from mtah.hibrido.decodificar import PASO, Costes, decodificar  # noqa: E402
from mtah.partido import A, D, H, X  # noqa: E402
from mtah.rapido import spell_f1  # noqa: E402
from mtah.rutas import cache_dir  # noqa: E402

FPS = 25
CLASES = ("ok", "borde", "fundida", "partida", "equipo", "perdida", "otro")
IOU_OK, IOU_BORDE = 0.5, 0.3
MAYORIA = 0.5
CUBRE_OTRA = 0.5
TROZO_MIN = 0.10
CUBOS_DUR = ((0, 3, "<3s"), (3, 8, "3-8s"), (8, 20, "8-20s"), (20, np.inf, ">20s"))
CUBOS_VIS = ((0, 0.5, "<0.5"), (0.5, 0.8, "0.5-0.8"), (0.8, 0.95, "0.8-0.95"), (0.95, 1.01, ">=0.95"))
ORACULOS = ("base", "vivo", "equipo", "ambos")
NEG = -1e6


def _emparejadas(over, umbral):
    """GT emparejadas en el voraz oficial (-iou, i_gt, j_pred) con IoU ≥ umbral."""
    usados_g, usados_p = set(), set()
    for iou, i, j in sorted(over, key=lambda t: (-t[0], t[1], t[2])):
        if iou < umbral:
            break
        if i in usados_g or j in usados_p:
            continue
        usados_g.add(i)
        usados_p.add(j)
    return usados_g


def clasificar(gt, pred, runs=None):
    """[{a, b, k, clase, iou}] por posesión GT."""
    gt = np.asarray(gt, np.int8)
    if runs is None:
        runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(pred, runs)
    ps = np.array([s[0] for s in p], np.int64)
    pe = np.array([s[1] for s in p], np.int64)
    gs = np.array([s[0] for s in g], np.int64)
    ge = np.array([s[1] for s in g], np.int64)
    over, por_g = [], [[] for _ in g]   # por_g[i] = [(iou, inter, j)] del mismo equipo
    for i, (a, b, k) in enumerate(g):
        j0, j1 = int(np.searchsorted(pe, a, side="right")), int(np.searchsorted(ps, b, side="left"))
        for j in range(j0, j1):
            if p[j][2] != k:
                continue
            inter = min(b, pe[j]) - max(a, ps[j])
            iou = inter / float(max(b, pe[j]) - min(a, ps[j]))
            over.append((iou, i, j))
            por_g[i].append((iou, inter, j))
    ok = _emparejadas(over, IOU_OK)
    filas = []
    for i, (a, b, k) in enumerate(g):
        n = b - a
        seg = pred[a:b]
        f_otro = float(np.mean(seg == (A if k == H else H)))
        f_d = float(np.mean(seg == D))
        mejor = max((t[0] for t in por_g[i]), default=0.0)
        if i in ok:
            clase = "ok"
        elif IOU_BORDE <= mejor < IOU_OK:
            clase = "borde"
        elif f_otro > MAYORIA:
            clase = "equipo"
        elif f_d > MAYORIA:
            clase = "perdida"
        elif _fundida(i, por_g[i], ps, pe, gs, ge):
            clase = "fundida"
        elif sum(inter >= TROZO_MIN * n for _, inter, _ in por_g[i]) >= 2:
            clase = "partida"
        else:
            clase = "otro"
        filas.append({"a": a, "b": b, "k": k, "clase": clase, "iou": mejor})
    return filas


def _fundida(i, cands, ps, pe, gs, ge):
    if not cands:
        return False
    j = max(cands, key=lambda t: (t[1], -t[2]))[2]
    a, b = ps[j], pe[j]
    i0, i1 = int(np.searchsorted(ge, a, side="right")), int(np.searchsorted(gs, b, side="left"))
    for o in range(i0, i1):
        if o != i and min(b, ge[o]) - max(a, gs[o]) >= CUBRE_OTRA * (ge[o] - gs[o]):
            return True
    return False


def logp_oraculo(logp, gt, modo):
    """logp [S, 3] con la información GT del oráculo `modo` (ver docstring del módulo)."""
    out = np.array(logp, np.float64, copy=True)
    if modo == "base":
        return out
    s = len(out)
    g = np.full(s * PASO, X, np.int8)
    g[:min(len(gt), s * PASO)] = gt[:s * PASO]
    cuenta = np.stack([(g.reshape(s, PASO) == c).sum(1) for c in (H, A, D)], 1)
    con = cuenta.sum(1) > 0
    vivo = con & (cuenta[:, 0] + cuenta[:, 1] > cuenta[:, 2])
    muerto = con & ~vivo
    if modo in ("vivo", "ambos"):
        out[muerto, :2] = NEG
        out[vivo, 2] = NEG
    if modo in ("equipo", "ambos"):
        es_h = cuenta[:, 0] >= cuenta[:, 1]
        out[vivo & es_h, 1] = NEG
        out[vivo & ~es_h, 0] = NEG
    return out


def _cubo(valor, cubos):
    for lo, hi, nombre in cubos:
        if lo <= valor < hi:
            return nombre
    return cubos[-1][2]


def items(oof, escenarios, conv, minimo_s, fuente_datos):
    """Genera (partido, semilla, escenario, gt, seccion, vis_paso, logp) de los partidos dev.

    Igual que afinar.items_base, pero solo lee gt, seccion_frames y balon de la caché (sin `jug`,
    por memoria) y nunca crea cachés.
    """
    fuentes = AF._fuentes_datos(fuente_datos)
    multi = len(fuentes) > 1
    for fuente_nombre in fuentes:
        fuente = AF._fuente_mod(fuente_nombre)
        for esc in escenarios:
            extra = AF._extra_fuente(fuente_nombre, ESCENARIOS[esc])
            for m0 in fuente.disponibles():
                m = AF._partido_item(fuente_nombre, m0, multi)
                if AF.en_caja(m) or AF.en_caja("{}:{}".format(fuente_nombre, m0)):
                    continue
                for s in AF.SEMILLAS:
                    ruta = os.path.join(cache_dir(), "rasgos_{}_{}_s{}_{}.npz".format(
                        fuente_nombre, m0, s, AF.version(extra)))
                    with np.load(ruta) as z:
                        gt0, sec, vis = z["gt"], z["seccion_frames"], z["balon"][:, 0].astype(np.float32)
                    gt = AF._gt(gt0, sec, conv, minimo_s) if fuente_nombre == "idsse" else gt0
                    with np.load(AF._oof_path(oof, m, s, esc)) as z:
                        logp = z["logp"]
                    yield m, s, esc, gt, sec, vis, logp


def _tabla(filas, clave, cubos):
    nombres = [c[2] for c in cubos]
    cab = "{:>10} {:>6} ".format(clave, "n") + " ".join("{:>8}".format(c) for c in CLASES)
    lineas = [cab]
    for nombre in nombres + ["total"]:
        sel = [f for f in filas if nombre == "total" or f[clave] == nombre]
        c = Counter(f["clase"] for f in sel)
        n = len(sel)
        lineas.append("{:>10} {:>6} ".format(nombre, n) + " ".join(
            "{:>8.3f}".format(c[k] / n if n else 0.0) for k in CLASES))
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", help="carpeta del run con oof_<partido>_s<semilla>[_<esc>].npz")
    ap.add_argument("--costes", required=True, help="json con 'costes' (o 'costes_final')")
    ap.add_argument("--fuente_datos", choices=("idsse", "metrica", "skillcorner", "estilo"),
                    default="idsse")
    ap.add_argument("--escenarios", default="v2")
    ap.add_argument("--convencion", choices=("dfl", "metrica"), default="dfl")
    ap.add_argument("--minimo_s", type=float, default=MINIMO_S_DEFECTO)
    ap.add_argument("--salida", default=None, help="json con los agregados")
    args = ap.parse_args()
    c = json.load(open(args.costes))
    costes = Costes(**c.get("costes_final", c.get("costes", c)))
    escenarios = [e.strip() for e in args.escenarios.split(",") if e.strip()]

    filas, spell, n_pred, partidos = [], {o: [] for o in ORACULOS}, 0, set()
    spell_esc = {}
    for m, s, esc, gt, sec, vis, logp in items(args.oof, escenarios, args.convencion,
                                                args.minimo_s, args.fuente_datos):
        partidos.add(m)
        runs = pm.valid_runs(gt)
        for o in ORACULOS:
            pred = decodificar(logp_oraculo(logp, gt, o), sec, costes)
            v = spell_f1(gt, pred, runs)
            spell[o].append(v)
            spell_esc.setdefault((esc, o), []).append(v)
            if o == "base":
                n_pred += len(pm.possession_spells(pm.fill_abstentions(pred, runs), runs))
                for f in clasificar(gt, pred, runs):
                    vf = vis[np.arange(f["a"], f["b"]) // PASO].mean()
                    filas.append({"clase": f["clase"],
                                  "dur": _cubo((f["b"] - f["a"]) / FPS, CUBOS_DUR),
                                  "vis": _cubo(float(vf), CUBOS_VIS)})

    print("run {}  fuente {}  escenarios {}  convención {}".format(
        args.oof, args.fuente_datos, ",".join(escenarios), args.convencion))
    print("costes {}".format(costes))
    print("partidos dev {}  items {}  posesiones GT {}  pred/gt {:.2f}".format(
        len(partidos), len(spell["base"]), len(filas), n_pred / max(len(filas), 1)))
    print("\nfracción de posesiones GT por clase, por duración")
    print(_tabla(filas, "dur", CUBOS_DUR))
    print("\nfracción de posesiones GT por clase, por balón visible")
    print(_tabla(filas, "vis", CUBOS_VIS))
    print("\noráculos (spellF1 medio por item)")
    print("{:>8} ".format("esc") + " ".join("{:>8}".format(o) for o in ORACULOS))
    for e in escenarios:
        print("{:>8} ".format(e) + " ".join(
            "{:>8.4f}".format(np.mean(spell_esc[(e, o)])) for o in ORACULOS))
    print("{:>8} ".format("todos") + " ".join("{:>8.4f}".format(np.mean(spell[o])) for o in ORACULOS))
    if args.salida:
        agregado = {
            "oof": args.oof, "fuente_datos": args.fuente_datos, "escenarios": escenarios,
            "costes": c.get("costes_final", c.get("costes", c)), "n_gt": len(filas), "n_pred": n_pred,
            "spell_f1": {o: float(np.mean(v)) for o, v in spell.items()},
            "spell_f1_por_escenario": {e: {o: float(np.mean(spell_esc[(e, o)])) for o in ORACULOS}
                                       for e in escenarios},
            "clases": {cl: {cb: sum(1 for f in filas if f[cl_k] == cb and f["clase"] == k)
                            for cb in [x[2] for x in cubos]}
                       for cl_k, cubos in (("dur", CUBOS_DUR), ("vis", CUBOS_VIS))
                       for k in CLASES for cl in [cl_k + ":" + k]},
        }
        json.dump(agregado, open(args.salida, "w"), indent=1)


if __name__ == "__main__":
    main()
