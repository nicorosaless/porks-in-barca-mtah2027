import numpy as np

from mtah.evaluar import pm
from mtah.hibrido.decodificar import PASO, Costes, decodificar, viterbi
from mtah.partido import A, D, H, X
from mtah.rapido import spell_f1


def _bloques(rng, n, largo):
    v = rng.integers(0, 3, n // largo + 1).astype(np.int8)
    return np.repeat(v, largo)[:n]


def test_spell_f1_rapido_igual_que_oficial():
    rng = np.random.default_rng(0)
    n = 20000
    gt = _bloques(rng, n, 113)
    gt[9000:9025] = X  # media parte
    runs = pm.valid_runs(gt)
    for largo in (7, 50, 113, 400):
        pred = _bloques(rng, n, largo)
        pred[rng.random(n) < 0.05] = X
        oficial = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
        assert spell_f1(gt, pred) == oficial


def test_viterbi_sin_coste_es_argmax_y_con_coste_no_parpadea():
    rng = np.random.default_rng(1)
    logp = rng.normal(size=(500, 3))
    assert (viterbi(logp, Costes(0, 0, 0)) == logp.argmax(1)).all()
    est = viterbi(logp, Costes(50, 50, 50))
    assert len(np.unique(est)) == 1


def test_decodificar_respeta_secciones():
    t = 1000
    sec = np.zeros(t, np.int8)
    sec[100:500] = 1
    sec[525:1000] = 2
    logp = np.zeros(((t - 1) // PASO + 1, 3))
    logp[:, H] = 1.0
    out = decodificar(logp, sec, Costes())
    assert (out[:100] == D).all() and (out[500:525] == D).all()
    assert (out[100:500] == H).all() and (out[525:] == H).all()
    assert A not in out


def test_afinar_ajusta_un_unico_coste_con_varios_escenarios(monkeypatch):
    from scripts.hibrido import afinar

    sec = np.ones(40, np.int8)
    gt = np.repeat(np.array([H, A, D, H], np.int8), 10)
    logp = np.full((8, 3), -5.0)
    logp[:, H] = 1.0
    afinar.ITEMS = [
        ("p1", 0, "v2", {"logp": logp}, gt, sec),
        ("p1", 0, "pes", {"logp": logp}, gt, sec),
        ("p2", 0, "v2", {"logp": logp}, gt, sec),
        ("p2", 0, "pes", {"logp": logp}, gt, sec),
    ]
    afinar.FUENTE = "run"
    fuera, cfg, _, filas = afinar._fold("p1")
    assert fuera == "p1"
    assert isinstance(cfg[1], Costes)
    assert {f[2] for f in filas} == {"v2", "pes"}


def test_afinar_kfold_estratifica_por_fuente():
    from scripts.hibrido import afinar

    folds = afinar._folds_k(
        ["idsse:b", "idsse:a", "metrica:SG2", "metrica:SG1", "skillcorner:2", "skillcorner:1"],
        2,
    )

    assert folds == [
        ("k0", ["idsse:a", "metrica:SG1", "skillcorner:1"]),
        ("k1", ["idsse:b", "metrica:SG2", "skillcorner:2"]),
    ]


def test_red_admite_bandera_de_convencion():
    import torch

    from mtah.hibrido.modelo import Red
    from mtah.hibrido.rasgos import FB, FJ

    red = Red(d=8, dt=8, capas_set=1, capas_t=1, convenciones=2)
    balon = torch.zeros(2, 5, FB)
    jug = torch.zeros(2, 5, 3, FJ)
    conv = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    logits, aux = red(balon, jug, conv)
    assert logits.shape == (2, 5, 3)
    assert aux.shape == (2, 5, 2)


def test_red_sin_bandera_mantiene_interfaz_antigua():
    import torch

    from mtah.hibrido.modelo import Red
    from mtah.hibrido.rasgos import FB, FJ

    red = Red(d=8, dt=8, capas_set=1, capas_t=1)
    logits, _ = red(torch.zeros(1, 4, FB), torch.zeros(1, 4, 2, FJ))
    assert logits.shape == (1, 4, 3)
