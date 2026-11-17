#!/usr/bin/env python3
"""Ajusta y evalúa el híbrido en IDSSE degradado con validación cruzada anidada.

Para cada partido i: busca parámetros (ascenso por coordenadas) maximizando el spellF1 medio en
los otros 6 partidos y puntúa el partido i con el scorer oficial. La media sobre los 7 es la
cifra honesta: el partido evaluado nunca ha intervenido en elegir sus parámetros. Además se
ajusta con los 7 a la vez: esos son los parámetros que se usarían en el reto.

Emisiones:
    heuristica          mtah/hibrido/heuristica.py (se ajustan pesos y costes)
    <carpeta de run>    log P de entrenar.py (oof_<partido>_s<semilla>.npz; solo costes)

    python scripts/hibrido/afinar.py heuristica
    python scripts/hibrido/afinar.py runs/it2
"""
import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass, fields, replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mtah.degradar import Degradacion, degradar  # noqa: E402
from mtah.convencion import MINIMO_S_DEFECTO, posesion_controlada  # noqa: E402
from mtah.evaluar import pm, puntuar  # noqa: E402
from mtah.escenarios import ESCENARIOS, PESIMISTA, SUFIJO  # noqa: E402
from mtah.fuentes import idsse, metrica, skillcorner  # noqa: E402
from mtah.hibrido import heuristica as HE  # noqa: E402
from mtah.hibrido import rasgos as R  # noqa: E402
from mtah.hibrido.decodificar import Costes, decodificar  # noqa: E402
from mtah.rapido import spell_f1  # noqa: E402
from mtah.rutas import cache_dir  # noqa: E402

SEMILLAS = (0, 1)


def version(extra):
    """La cache depende de la degradación: si cambian sus parámetros, cambia el nombre."""
    return "{:08x}".format(zlib.crc32(repr(Degradacion(**extra)).encode()))
REJILLA = {  # valores candidatos por parámetro
    "cambio": [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    "a_muerto": [1, 2, 4, 6, 8, 12, 16, 24, 32],
    "desde_muerto": [1, 2, 4, 6, 8, 12, 16, 24, 32],
    "sesgo_d": [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2],
    "temperatura": [0.25, 0.5, 0.75, 1, 1.5, 2, 3],
    # E3 (duración mínima, en pasos de 0,2 s): apagado salvo --duraciones
    "alfa": [0, 0.5, 1, 2],
    "beta": [0, 0.5, 1, 2],
    "gamma": [0, 0.5, 1, 2],
    "min_pos": [0],
    "min_d": [0],
    "tau": [0.5, 1, 1.5, 2, 3, 4, 6],
    "base_d": [-6, -5, -4, -3, -2, -1, 0],
    "fuera": [0, 2, 4, 6, 10],
    "falta": [-1, 0, 0.5, 1, 2, 3],
    "hueco": [0, 0.25, 0.5, 1, 1.5, 2],
    "lento": [0, 0.5, 1, 2],
}


def _escenarios_arg(valor):
    return [x.strip() for x in valor.split(",") if x.strip()]


def _gt(gt, seccion, conv, minimo_s):
    if conv != "metrica":
        return gt
    return posesion_controlada(gt, seccion, int(round(minimo_s * 25)))


def _extra_fuente(fuente, extra):
    if fuente in ("metrica", "skillcorner"):
        return {**extra, "balon_ausente": 0.0, "quitar_congelado": False}
    return extra


def _fuente_mod(nombre):
    if nombre == "metrica":
        return metrica
    if nombre == "skillcorner":
        return skillcorner
    return idsse


def _fuentes_datos(nombre):
    if nombre == "estilo":
        return ["metrica", "skillcorner"]
    return [x.strip() for x in nombre.replace("+", ",").split(",") if x.strip()]


def _partido_item(fuente, partido, multi):
    return "{}:{}".format(fuente, partido) if multi else partido


def _split_item(m, fuente_defecto):
    if ":" in m:
        return m.split(":", 1)
    return fuente_defecto, m


def _oof_path(fuente, m, s, esc):
    suf = SUFIJO[esc]
    fuente_m, partido_m = _split_item(m, "")
    candidatos = [
        os.path.join(fuente, "oof_{}_s{}{}.npz".format(m, s, suf)),
        os.path.join(fuente, "oof_idsse:{}_s{}{}.npz".format(partido_m, s, suf)),
        os.path.join(fuente, "oof_metrica:{}_s{}{}.npz".format(partido_m, s, suf)),
        os.path.join(fuente, "oof_skillcorner:{}_s{}{}.npz".format(partido_m, s, suf)),
    ]
    if fuente_m:
        candidatos.insert(1, os.path.join(fuente, "oof_{}:{}_s{}{}.npz".format(
            fuente_m, partido_m, s, suf)))
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    return candidatos[0]


def items_base(escenarios=("v2",), conv="dfl", minimo_s=MINIMO_S_DEFECTO, fuente_datos="idsse"):
    """[(partido, semilla, escenario, rasgos, gt, seccion)] con cache local."""
    out = []
    fuentes = _fuentes_datos(fuente_datos)
    multi = len(fuentes) > 1
    for fuente_nombre in fuentes:
        fuente = _fuente_mod(fuente_nombre)
        for esc in escenarios:
            extra = _extra_fuente(fuente_nombre, ESCENARIOS[esc])
            for m0 in fuente.disponibles():
                m = _partido_item(fuente_nombre, m0, multi)
                p = None
                for s in SEMILLAS:
                    ruta = os.path.join(cache_dir(), "rasgos_{}_{}_s{}_{}.npz".format(
                        fuente_nombre, m0, s, version(extra)))
                    if not os.path.exists(ruta):
                        p = p or fuente.cargar(m0)
                        r = R.rasgos(degradar(p, Degradacion(semilla=s, **extra)))
                        np.savez_compressed(ruta, gt=p.gt, seccion_frames=p.seccion, **r)
                    z = dict(np.load(ruta))
                    gt = _gt(z["gt"], z["seccion_frames"], conv, minimo_s) if fuente_nombre == "idsse" else z["gt"]
                    out.append((m, s, esc, z, gt,
                                z["seccion_frames"]))
    return out


ITEMS = None
FUENTE = None
COSTES_FIJOS = None
TAGS = None  # E4: None, o el nombre del ruido con que se simulan los tags de ajuste


@dataclass(frozen=True)
class Tags:
    """Pesos del sesgo del Smart Tagging (mtah/hibrido/sesgo_tags.py) que ajusta el ascenso."""
    alfa: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


# Ruidos de la simulación de tags (fijados en docs/HIBRIDO.md antes de evaluar).
RUIDOS_TAGS = {
    "nominal": dict(),
    "pesimista": dict(jitter_s=3.0, volteo=0.3, cobertura=0.8),
    "basura": dict(volteo=0.5),
}


def _tags_item(m, s, ruido):
    from mtah.simular_tags import Ruido, simular
    fuente_m, partido_m = _split_item(m, "idsse")
    return simular(fuente_m, partido_m, semilla=s, ruido=Ruido(**RUIDOS_TAGS[ruido]))
# Caja fuerte de it9 (docs/HIBRIDO.md): i % 3 == 2 por fuente sobre ids ordenados. Con --dev se excluye.
CAJA_FUERTE = {"idsse:J03WOH", "idsse:J03WQQ", "metrica:SG3", "skillcorner:1899585",
               "skillcorner:1953632", "skillcorner:1996435", "skillcorner:2006363",
               "skillcorner:2010085", "skillcorner:2015213"}


def en_caja(m):
    return m in CAJA_FUERTE or "idsse:" + m in CAJA_FUERTE


def _init(fuente, escenarios=("v2",), conv="dfl", minimo_s=MINIMO_S_DEFECTO, fuente_datos="idsse",
          solo=None, costes_fijos=None, duraciones=False, tags=None, tags_eval=None):
    global ITEMS, FUENTE, COSTES_FIJOS, TAGS
    TAGS = tags
    if duraciones:
        REJILLA["min_pos"] = [0, 2, 3, 5, 8, 10, 15]
        REJILLA["min_d"] = [0, 3, 5, 10, 15, 20, 25]
    FUENTE = fuente
    COSTES_FIJOS = costes_fijos
    ITEMS = items_base(escenarios, conv, minimo_s, fuente_datos)
    if solo is not None:
        ITEMS = [it for it in ITEMS if it[0] in set(solo)]
    if fuente != "heuristica":
        nuevos = []
        for m, s, esc, z, gt, sec in ITEMS:
            logp = np.load(_oof_path(fuente, m, s, esc))["logp"]
            z = {"logp": logp}
            if tags:
                z["tags"] = _tags_item(m, s, tags)
                z["tags_eval"] = _tags_item(m, s, tags_eval or tags)
            nuevos.append((m, s, esc, z, gt, sec))
        ITEMS = nuevos


def predecir(z, sec, cfg, evaluar=False):
    pesos, costes = cfg
    logp = z["logp"] if "logp" in z else HE.emisiones(z, pesos)
    if isinstance(pesos, Tags) and "tags" in z and (pesos.alfa or pesos.beta or pesos.gamma):
        from mtah.hibrido.sesgo_tags import ParamsSesgo, sesgo
        inst = z["tags_eval"] if evaluar else z["tags"]
        logp = logp + sesgo(inst, len(logp), params=ParamsSesgo(
            alfa=pesos.alfa, beta=pesos.beta, gamma=pesos.gamma), seccion=sec)
    return decodificar(logp, sec, costes)


def objetivo(partidos, cfg):
    sel = [it for it in ITEMS if it[0] in partidos]
    return float(np.mean([spell_f1(gt, predecir(z, sec, cfg)) for _, _, _, z, gt, sec in sel]))


def _con(cfg, nombre, valor):
    pesos, costes = cfg
    if nombre in {f.name for f in fields(Costes)}:
        return pesos, replace(costes, **{nombre: valor})
    return replace(pesos, **{nombre: valor}), costes


def ascenso(partidos, rondas=2):
    if COSTES_FIJOS is not None:
        cfg = (Tags() if TAGS else HE.Pesos(), Costes(**COSTES_FIJOS))
        return cfg, objetivo(partidos, cfg)
    cfg = (Tags() if TAGS else HE.Pesos(), Costes())
    nombres = [f.name for f in fields(Costes)]
    if TAGS:
        nombres = nombres + [f.name for f in fields(Tags)]
    if FUENTE == "heuristica":
        nombres = [f.name for f in fields(HE.Pesos)] + nombres
    mejor = objetivo(partidos, cfg)
    for _ in range(rondas):
        antes = mejor
        for n in nombres:
            for v in REJILLA[n]:
                c = _con(cfg, n, float(v))
                o = objetivo(partidos, c)
                if o > mejor + 1e-6:
                    mejor, cfg = o, c
        if mejor - antes < 1e-4:
            break
    return cfg, mejor


def _fold(fuera):
    todos = sorted({it[0] for it in ITEMS})
    if isinstance(fuera, tuple):
        nombre, validacion = fuera
    else:
        nombre, validacion = fuera, ([fuera] if fuera is not None else [])
    train = [m for m in todos if m not in set(validacion)]
    cfg, o_train = ascenso(train)
    filas = []
    if validacion:
        for m, s, esc, z, gt, sec in ITEMS:
            if m in set(validacion):
                r = puntuar(gt, predecir(z, sec, cfg, evaluar=True))
                filas.append((m, s, esc, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                              r["n_gt_spells"]))
    return nombre, cfg, o_train, filas


def _folds_k(todos, k):
    if not k:
        return todos
    grupos = [[] for _ in range(k)]
    for fuente in sorted({_split_item(m, "idsse")[0] for m in todos}):
        propios = sorted(m for m in todos if _split_item(m, "idsse")[0] == fuente)
        for i, m in enumerate(propios):
            grupos[i % k].append(m)
    return [("k{}".format(i), g) for i, g in enumerate(grupos) if g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fuente")
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--sufijo", default="", help="_pes para el escenario pesimista")
    ap.add_argument("--escenarios", default=None, help="v2,pes por ejemplo")
    ap.add_argument("--convencion", choices=("dfl", "metrica"), default="dfl")
    ap.add_argument("--minimo_s", type=float, default=MINIMO_S_DEFECTO)
    ap.add_argument("--fuente_datos", choices=("idsse", "metrica", "skillcorner", "estilo"),
                    default="idsse")
    ap.add_argument("--kfold", type=int, default=0,
                    help="K folds deterministas por id; 0 = leave-one-match-out")
    ap.add_argument("--salida", default=None, help="json con resultados")
    ap.add_argument("--dev", action="store_true", help="excluye la caja fuerte de it9")
    ap.add_argument("--solo_caja", action="store_true", help="evalúa SOLO la caja fuerte (corte final)")
    ap.add_argument("--tags", choices=sorted(RUIDOS_TAGS), default=None,
                    help="E4: simula Smart Tagging con este ruido para ajustar alfa/beta/gamma")
    ap.add_argument("--tags_eval", choices=sorted(RUIDOS_TAGS), default=None,
                    help="ruido de los tags al puntuar la validación (por defecto el de --tags)")
    ap.add_argument("--duraciones", action="store_true", help="E3: ajusta también min_pos y min_d")
    ap.add_argument("--costes_fijos", default=None,
                    help="json de costes (o fichero .json con 'costes'/'costes_final'); no ajusta")
    args = ap.parse_args()
    costes_fijos = None
    if args.costes_fijos:
        c = json.load(open(args.costes_fijos)) if os.path.exists(args.costes_fijos) else json.loads(args.costes_fijos)
        costes_fijos = c.get("costes_final", c.get("costes", c))
    t0 = time.time()
    escenarios = _escenarios_arg(args.escenarios) if args.escenarios else (["pes"] if args.sufijo == "_pes" else ["v2"])
    for e in escenarios:
        if e not in ESCENARIOS:
            raise SystemExit("escenario desconocido {} (disponibles: {})".format(e, sorted(ESCENARIOS)))
    items_base(escenarios, args.convencion, args.minimo_s, args.fuente_datos)  # crea la cache antes de repartir
    if args.fuente_datos == "estilo":
        todos = ["metrica:" + m for m in metrica.disponibles()] + [
            "skillcorner:" + m for m in skillcorner.disponibles()]
    else:
        todos = _fuente_mod(args.fuente_datos).disponibles()
    if args.dev:
        todos = [m for m in todos if not en_caja(m)]
    elif args.solo_caja:
        todos = [m for m in todos if en_caja(m)]
    with Pool(args.procesos, initializer=_init,
              initargs=(args.fuente, tuple(escenarios), args.convencion, args.minimo_s,
                        args.fuente_datos, todos, costes_fijos, args.duraciones, args.tags,
                        args.tags_eval)) as pool:
        res = pool.map(_fold, [*_folds_k(todos, args.kfold), None])
    filas = [f for _, _, _, fs in res for f in fs]
    spell = np.array([f[3] for f in filas])
    macro = np.array([f[4] for f in filas])
    por_partido = {m: np.mean([f[3] for f in filas if f[0] == m]) for m in todos}
    por_escenario = {e: np.mean([f[3] for f in filas if f[2] == e]) for e in escenarios}
    macro_por_escenario = {e: np.mean([f[4] for f in filas if f[2] == e]) for e in escenarios}
    final = [r for r in res if r[0] is None][0]
    print("\nfuente {}  ({:.0f} s)".format(args.fuente, time.time() - t0))
    for m in todos:
        print("  {}  spellF1 {:.4f}".format(m, por_partido[m]))
    for e in escenarios:
        print("  escenario {}  spellF1 {:.4f} macroF1 {:.4f}".format(
            e, por_escenario[e], macro_por_escenario[e]))
    print("CV anidada: spellF1 {:.4f} (min {:.3f} max {:.3f})  macroF1 {:.4f}  pred/gt {:.2f}".format(
        spell.mean(), min(por_partido.values()), max(por_partido.values()), macro.mean(),
        sum(f[5] for f in filas) / sum(f[6] for f in filas)))
    print("parámetros con todos los partidos (spellF1 en ajuste {:.4f}):".format(final[2]))
    print("  ", final[1][1])
    if args.fuente == "heuristica":
        print("  ", final[1][0])
    if args.salida:
        json.dump({"fuente": args.fuente, "spell_f1": spell.mean(), "macro_f1": macro.mean(),
                   "spell_f1_por_escenario": por_escenario,
                   "macro_f1_por_escenario": macro_por_escenario,
                   "por_partido": por_partido, "filas": filas,
                   "costes_final": asdict(final[1][1]), "pesos_final": asdict(final[1][0]),
                   "folds": {str(r[0]): asdict(r[1][1]) for r in res if r[0] is not None}},
                  open(args.salida, "w"), indent=1)


if __name__ == "__main__":
    main()
