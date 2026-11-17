"""spellF1 rápido para afinar: mismo resultado que evaluation/possession_metrics.spell_scores.

El oficial compara todas las parejas de posesiones (n_gt x n_pred bucles en Python, ~2 s por
partido). Aquí solo se miran las parejas que se solapan, en el mismo orden de desempate
(-iou, i_gt, j_pred), así que el emparejamiento voraz es idéntico. tests/test_hibrido.py lo
compara con el oficial. Las cifras que se reportan salen siempre del scorer oficial.
"""
import numpy as np

from mtah.evaluar import pm


def _spells(codes, runs):
    return pm.possession_spells(codes, runs)


def spell_f1(gt, pred, runs=None):
    gt = np.asarray(gt, np.int8)
    pred = np.asarray(pred, np.int8)
    if runs is None:
        runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, runs)
    g = _spells(gt, runs)
    p = _spells(pred, runs)
    if not g or not p:
        return 0.0
    ps = np.array([s[0] for s in p])
    pe = np.array([s[1] for s in p])
    pk = np.array([s[2] for s in p])
    over = []
    for i, (a, b, k) in enumerate(g):
        # posesiones predichas que empiezan antes de b y acaban después de a (están ordenadas)
        j0 = int(np.searchsorted(pe, a, side="right"))
        j1 = int(np.searchsorted(ps, b, side="left"))
        for j in range(j0, j1):
            if pk[j] != k:
                continue
            inter = min(b, pe[j]) - max(a, ps[j])
            union = max(b, pe[j]) - min(a, ps[j])
            over.append((inter / float(union), i, j))
    over.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for th in pm.SPELL_IOUS:
        usados_g, usados_p, tp = set(), set(), 0
        for iou, i, j in over:
            if iou < th:
                break
            if i in usados_g or j in usados_p:
                continue
            usados_g.add(i)
            usados_p.add(j)
            tp += 1
        prec = tp / len(p)
        rec = tp / len(g)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))
