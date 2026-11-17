#!/usr/bin/env python3
"""Entrena el modelo de emisiones en IDSSE degradado, dejando un partido fuera cada vez.

Solo datos externos (IDSSE): se ejecuta en MareNostrum. Para cada partido i:
  - entrena con los otros 6, cada uno con `--semillas` degradaciones distintas (semillas 1000+),
  - predice el partido i con las degradaciones de evaluación (semillas 0 y 1, las mismas que
    usa scripts/evaluar_externo.py) y guarda log P(H, A, D) por paso.
Con --final, entrena además con los 7 y guarda los pesos (para predecir el reto en local).

Épocas fijas: no hay parada temprana mirando el partido que se deja fuera.

    python scripts/hibrido/entrenar.py --run it2 --salida $HOME/mtah2027/runs
"""
import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mtah.degradar import Degradacion, degradar  # noqa: E402
from mtah.convencion import MINIMO_S_DEFECTO, posesion_controlada  # noqa: E402
from mtah.escenarios import ESCENARIOS, PESIMISTA, SUFIJO  # noqa: E402
from mtah.fuentes import idsse, metrica, skillcorner  # noqa: E402
from mtah.hibrido import rasgos as R  # noqa: E402

SEMILLAS_EVAL = (0, 1)


# Con --mezcla, las degradaciones de entrenamiento rotan entre tres niveles de "el balón muerto
# falta": el por defecto, uno intermedio y el pesimista.
NIVELES = [dict(), dict(congelado_ausente=0.35, congelado_fuera_ausente=0.45), PESIMISTA]


def _partido_fuente(clave):
    if ":" not in clave:
        return "idsse", clave
    return tuple(clave.split(":", 1))


def _cargar_partido(clave):
    fuente, partido = _partido_fuente(clave)
    if fuente == "idsse":
        return idsse.cargar(partido)
    if fuente == "metrica":
        return metrica.cargar(partido)
    if fuente == "skillcorner":
        return skillcorner.cargar(partido)
    raise ValueError("fuente desconocida: {}".format(fuente))


def _convencion_partido(clave):
    return 1 if _partido_fuente(clave)[0] in ("metrica", "skillcorner") else 0


def _extra_fuente(clave, extra):
    # "balon_ausente_estilo": huecos de balón extra (con el juego vivo o no) en Metrica/SkillCorner,
    # para que la red no aprenda "sin balón = muerto" de las repeticiones de la retransmisión.
    extra = dict(extra)
    estilo = extra.pop("balon_ausente_estilo", 0.0)
    if _partido_fuente(clave)[0] in ("metrica", "skillcorner"):
        return {**extra, "balon_ausente": estilo, "quitar_congelado": False}
    return extra


def _preparar(args):
    m, clave, semilla, extra, conv, minimo_s = args
    p = _cargar_partido(m)
    if conv == "metrica" and _partido_fuente(m)[0] == "idsse":
        mf = int(round(minimo_s * p.fps))
        p = replace(p, gt=posesion_controlada(p.gt, p.seccion, mf))
    r = R.rasgos(degradar(p, Degradacion(semilla=int(semilla), **_extra_fuente(m, extra))),
                 limpio=p)
    return m, clave, r


def _escenarios_arg(valor):
    valor = valor.replace("+", ",")
    return [x.strip() for x in valor.split(",") if x.strip()]


def _folds_k(partidos, k):
    if not k:
        return [(m, [m]) for m in partidos]
    grupos = [[] for _ in range(k)]
    for fuente in sorted({_partido_fuente(m)[0] for m in partidos}):
        propios = sorted(m for m in partidos if _partido_fuente(m)[0] == fuente)
        for i, m in enumerate(propios):
            grupos[i % k].append(m)
    return [("k{}".format(i), g) for i, g in enumerate(grupos) if g]


def _peso_borde(etq, peso, radio):
    import torch

    valid = etq >= 0
    if peso == 1.0 or radio <= 0:
        return valid.float()
    borde = torch.zeros_like(valid)
    for d in range(1, radio + 1):
        dif = valid[:, d:] & valid[:, :-d] & (etq[:, d:] != etq[:, :-d])
        borde[:, d:] |= dif
        borde[:, :-d] |= dif
    return torch.where(borde & valid, torch.as_tensor(float(peso), device=etq.device), 1.0) * valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="partidos a dejar fuera (todos)")
    ap.add_argument("--kfold", type=int, default=0,
                    help="K folds estratificados por fuente e id; 0 = leave-one-match-out")
    ap.add_argument("--semillas", type=int, default=6)
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lotes_epoca", type=int, default=64)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--largo", type=int, default=512, help="pasos por ventana (5 fps)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.0, help="peso de la pérdida de balón")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dt", type=int, default=128)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--procesos", type=int, default=16)
    ap.add_argument("--pesimista", action="store_true", help="evalúa también el escenario pesimista")
    ap.add_argument("--escenarios_eval", default="v2",
                    help="escenarios separados por coma; --pesimista equivale a v2,pes")
    ap.add_argument("--convencion", choices=("dfl", "metrica"), default="dfl")
    ap.add_argument("--minimo_s", type=float, default=MINIMO_S_DEFECTO)
    ap.add_argument("--solo_eval", default=None, help="run previo con fold_<partido>.pt")
    ap.add_argument("--deg_train", choices=sorted(ESCENARIOS), default="v2")
    ap.add_argument("--peso_borde", type=float, default=1.0)
    ap.add_argument("--radio_borde", type=int, default=5)
    ap.add_argument("--mezcla", action="store_true", help="entrena con varios niveles de balón muerto ausente")
    ap.add_argument("--fuentes", default="idsse", help="fuentes separadas por coma: idsse,metrica")
    ap.add_argument("--peso_metrica", type=float, default=2.0)
    ap.add_argument("--peso_estilo", type=float, default=None,
                    help="peso de entrenamiento para Metrica y SkillCorner (por defecto peso_metrica)")
    ap.add_argument("--semilla_red", type=int, default=0,
                    help="desplazamiento de la semilla de la red (0 = la de siempre); para ensambles")
    ap.add_argument("--huecos_estilo", default="0",
                    help="niveles de balon_ausente para Metrica/SkillCorner en entrenamiento, "
                         "rotando por semilla (p. ej. 0+0.55+0.65); 0 = sin huecos extra")
    ap.add_argument("--bandera_convencion", action="store_true",
                    help="añade una entrada constante 0=IDSSE/DFL, 1=Metrica")
    args = ap.parse_args()

    import torch
    from torch import nn

    from mtah.hibrido.modelo import Red, aumentar, predecir, tramos_pasos

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(args.salida, args.run)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)
    fuentes = [x.strip() for x in args.fuentes.replace("+", ",").split(",") if x.strip()]
    partidos_base = []
    if "idsse" in fuentes:
        partidos_base += ["idsse:" + m for m in idsse.disponibles()]
    if "metrica" in fuentes:
        partidos_base += ["metrica:" + m for m in metrica.disponibles()]
    if "skillcorner" in fuentes:
        partidos_base += ["skillcorner:" + m for m in skillcorner.disponibles()]
    partidos = args.partidos or partidos_base
    partidos = [p if ":" in p else "idsse:" + p for p in partidos]
    t0 = time.time()
    escenarios_eval = ["v2", "pes"] if args.pesimista else _escenarios_arg(args.escenarios_eval)
    for e in [*escenarios_eval, args.deg_train]:
        if e not in ESCENARIOS:
            raise SystemExit("escenario desconocido {} (disponibles: {})".format(e, sorted(ESCENARIOS)))
    evals = [(s, e) for e in escenarios_eval for s in SEMILLAS_EVAL]
    huecos_estilo = [float(x) for x in _escenarios_arg(args.huecos_estilo)]

    def extra(s):
        if s < 1000:
            return {}
        e = dict(ESCENARIOS[args.deg_train])
        if args.mezcla:
            e.update(NIVELES[(s - 1000) % len(NIVELES)])
        e["balon_ausente_estilo"] = huecos_estilo[(s - 1000) % len(huecos_estilo)]
        return e
    tareas = []
    for m in partidos:
        for s, esc in evals:
            suf = SUFIJO[esc]
            tareas.append((m, "{}{}".format(s, suf), s, ESCENARIOS[esc], args.convencion, args.minimo_s))
        for s in range(1000, 1000 + args.semillas):
            tareas.append((m, s, s, extra(s), args.convencion, args.minimo_s))
    with Pool(args.procesos) as pool:
        datos = {(m, s): r for m, s, r in pool.map(_preparar, tareas)}
    print("rasgos: {} items en {:.0f} s".format(len(datos), time.time() - t0), flush=True)

    if args.solo_eval:
        prev = os.path.join(args.salida, args.solo_eval)
        from mtah.hibrido.modelo import Red, predecir
        for m in (args.folds or partidos):
            ck = torch.load(os.path.join(prev, "fold_{}.pt".format(m)), map_location=dev)
            red = Red(d=ck["args"]["d"], dt=ck["args"]["dt"],
                      convenciones=ck["args"].get("convenciones", 0)).to(dev)
            red.load_state_dict(ck["estado"])
            for s, esc in evals:
                clave = "{}{}".format(s, SUFIJO[esc])
                logp = predecir(red, datos[(m, clave)], dev, _convencion_partido(m))
                np.savez_compressed(os.path.join(out, "oof_{}_s{}.npz".format(m, clave)),
                                    logp=logp)
        print("solo_eval hecho en {:.0f} s".format(time.time() - t0), flush=True)
        return

    def a_gpu(r):
        real = r["balon_real"].copy()
        etiq = r["etiq"].astype(np.int64)
        return (torch.from_numpy(r["balon"]).to(dev), torch.from_numpy(r["jug"]).to(dev),
                torch.from_numpy(etiq).to(dev), torch.from_numpy(np.nan_to_num(real)).to(dev),
                torch.from_numpy(~np.isnan(real[:, 0]) & (r["balon"][:, 0] == 0)).to(dev))

    def entrenar(train_partidos, semilla_red):
        torch.manual_seed(semilla_red)
        gen = torch.Generator().manual_seed(semilla_red)
        items = [a_gpu(datos[(m, s)]) + (_convencion_partido(m),) for m in train_partidos
                 for s in range(1000, 1000 + args.semillas)]
        ventanas = []  # (item, ini, fin) de tramos con al menos `largo` pasos
        for i, (m, s) in enumerate((m, s) for m in train_partidos
                                   for s in range(1000, 1000 + args.semillas)):
            for a, b in tramos_pasos(datos[(m, s)]["seccion"]):
                if b - a >= args.largo:
                    peso_estilo = args.peso_metrica if args.peso_estilo is None else args.peso_estilo
                    peso = peso_estilo if _partido_fuente(m)[0] in ("metrica", "skillcorner") else 1.0
                    ventanas.append((i, a, b, peso))
        pesos = np.array([(b - a - args.largo + 1) * w for _, a, b, w in ventanas], float)
        pesos /= pesos.sum()
        rng = np.random.default_rng(semilla_red)
        red = Red(d=args.d, dt=args.dt, convenciones=2 if args.bandera_convencion else 0).to(dev)
        opt = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=1e-2)
        total = args.epocas * args.lotes_epoca
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.1)
        ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        idx = torch.arange(args.largo, device=dev)
        for paso in range(total):
            red.train()
            elegidas = rng.choice(len(ventanas), args.lote, p=pesos)
            lb, lj, le, lr_, lm, lc = [], [], [], [], [], []
            for v in elegidas:
                i, a, b, _ = ventanas[v]
                ini = int(rng.integers(a, b - args.largo + 1))
                bal, jug, etq, real, mreal, conv = items[i]
                sl = idx + ini
                lb.append(bal[sl]); lj.append(jug[sl]); le.append(etq[sl])
                lr_.append(real[sl]); lm.append(mreal[sl])
                lc.append(conv)
            bal, jug, etq, real = aumentar(torch.stack(lb), torch.stack(lj).float(),
                                           torch.stack(le), torch.stack(lr_), gen)
            mreal = torch.stack(lm)
            conv = torch.tensor(lc, dtype=torch.long, device=dev)[:, None].expand(-1, args.largo)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits, pb = red(bal, jug, conv)
            ce_paso = ce(logits.float().reshape(-1, 3), etq.reshape(-1)).reshape_as(etq)
            w = _peso_borde(etq, args.peso_borde, args.radio_borde)
            perdida = (ce_paso * w).sum() / w.sum().clamp(min=1.0)
            if args.aux > 0 and mreal.any():
                err = ((pb.float() - real) ** 2).sum(-1)
                perdida = perdida + args.aux * err[mreal].mean()
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step()
            sched.step()
            if (paso + 1) % (args.lotes_epoca * 5) == 0:
                print("  paso {}/{} pérdida {:.4f} ({:.0f} s)".format(
                    paso + 1, total, perdida.item(), time.time() - t0), flush=True)
        return red

    resumen = {}
    folds = [(m, [m]) for m in args.folds] if args.folds else _folds_k(partidos, args.kfold)
    for nombre_fold, validacion in folds:
        train = [x for x in partidos if x not in set(validacion)]
        print("fold {}: valido {} entreno con {}".format(nombre_fold, validacion, train), flush=True)
        red = entrenar(train, semilla_red=zlib.crc32(nombre_fold.encode()) % 1000 + 1000 * args.semilla_red)
        ck_args = dict(vars(args), convenciones=2 if args.bandera_convencion else 0)
        torch.save({"estado": red.state_dict(), "args": ck_args},
                   os.path.join(out, "fold_{}.pt".format(nombre_fold)))
        for m in validacion:
            for s, esc in evals:
                clave = "{}{}".format(s, SUFIJO[esc])
                r = datos[(m, clave)]
                logp = predecir(red, r, dev, _convencion_partido(m))
                np.savez_compressed(os.path.join(out, "oof_{}_s{}.npz".format(m, clave)), logp=logp)
                ok = r["etiq"] >= 0
                acc = float((logp.argmax(1)[ok] == r["etiq"][ok]).mean())
                resumen["{}_s{}".format(m, clave)] = acc
                print("  {} s{} acierto por paso {:.4f}".format(m, clave, acc), flush=True)
    if args.final:
        red = entrenar(partidos, semilla_red=12345 + args.semilla_red)
        ck_args = dict(vars(args), convenciones=2 if args.bandera_convencion else 0)
        torch.save({"estado": red.state_dict(), "args": ck_args},
                   os.path.join(out, "final.pt"))
    json.dump(resumen, open(os.path.join(out, "resumen.json"), "w"), indent=1)
    print("hecho en {:.0f} s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
