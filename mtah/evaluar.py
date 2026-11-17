"""Puntúa con el scorer oficial (evaluation/possession_metrics.py), sin reimplementar métricas."""
import os
import sys

import numpy as np

from mtah.rutas import RAIZ

sys.path.insert(0, os.path.join(RAIZ, "evaluation"))
import possession_metrics as pm  # noqa: E402

CAMPOS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_gt_alive",
          "n_pred_spells", "n_gt_spells")


def puntuar(gt, pred, bootstrap_n=0):
    r = pm.evaluate(np.asarray(gt, np.int8), np.asarray(pred, np.int8), bootstrap_n=bootstrap_n)
    return {c: r[c] for c in CAMPOS} | {"_completo": r}
