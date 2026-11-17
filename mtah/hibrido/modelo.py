"""Modelo de emisiones del híbrido: P(H, A, D) por paso a partir de los rasgos.

Dos etapas, siguiendo la literatura de inferencia de balón (Kim et al. KDD 2023, TranSPORTmer):

1. Conjunto por paso: un token por jugador visible y otro del balón, Transformer sin posición
   (invariante al orden de los slots, que en el ATD no significan nada). Se resume en el token
   del balón y la media de cada equipo.
2. Temporal: GRU bidireccional sobre la secuencia de resúmenes (usa pasado y futuro, #15).

Salidas: logits H/A/D por paso y, opcional, la posición real del balón (objetivo auxiliar que
obliga a aprender dónde está el balón cuando falta).
"""
import numpy as np
import torch
from torch import nn

from mtah.hibrido.rasgos import FB, FJ

# índices de rasgos que cambian de signo al reflejar el campo
JX, JY = [2, 4, 6], [3, 5, 7]
BX, BY = [1, 3, 9, 11], [2, 4, 10, 12]


def mlp(a, b):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))


class Red(nn.Module):
    def __init__(self, d=64, dt=128, capas_set=2, capas_t=2, dropout=0.1, convenciones=0):
        super().__init__()
        self.pj = mlp(FJ, d)
        self.pb = mlp(FB, d)
        self.conv = nn.Embedding(convenciones, d) if convenciones else None
        capa = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True,
                                          norm_first=True)
        self.set = nn.TransformerEncoder(capa, capas_set, enable_nested_tensor=False)
        self.proy = nn.Sequential(nn.Linear(3 * d, dt), nn.GELU())
        self.gru = nn.GRU(dt, dt, num_layers=capas_t, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.cab = nn.Linear(2 * dt, 3)
        self.aux = nn.Linear(2 * dt, 2)

    def forward(self, balon, jug, convencion=None):
        b, l, k, _ = jug.shape
        jf = jug.reshape(b * l, k, FJ)
        tok = torch.cat([self.pb(balon.reshape(b * l, 1, FB)), self.pj(jf)], 1)
        if self.conv is not None:
            if convencion is None:
                convencion = torch.zeros((b, l), dtype=torch.long, device=balon.device)
            if not torch.is_tensor(convencion):
                convencion = torch.full((b, l), int(convencion), dtype=torch.long, device=balon.device)
            c = self.conv(convencion.reshape(b * l)).unsqueeze(1)
            tok = tok + c
        vis = jf[..., 0] > 0
        pad = torch.cat([torch.zeros_like(vis[:, :1]), ~vis], 1)
        h = self.set(tok, src_key_padding_mask=pad)
        hj = h[:, 1:]
        loc = (jf[..., 1] > 0).unsqueeze(-1).float()
        vis_f = vis.unsqueeze(-1).float()
        ph = (hj * loc * vis_f).sum(1) / (loc * vis_f).sum(1).clamp(min=1)
        va = (1 - loc) * vis_f
        pa = (hj * va).sum(1) / va.sum(1).clamp(min=1)
        z = self.proy(torch.cat([h[:, 0], ph, pa], -1)).reshape(b, l, -1)
        y, _ = self.gru(z)
        return self.cab(y), self.aux(y)


def tramos_pasos(seccion_paso):
    """[(ini, fin)] de pasos contiguos con la misma sección > 0."""
    out, i, n = [], 0, len(seccion_paso)
    while i < n:
        if seccion_paso[i] > 0:
            j = i
            while j < n and seccion_paso[j] == seccion_paso[i]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


@torch.no_grad()
def predecir(red, r, device="cpu", convencion=0):
    """rasgos -> log P(H, A, D) [S, 3] (numpy). Cada parte se procesa entera, por separado."""
    import numpy as np
    red.eval()
    s = len(r["seccion"])
    out = np.full((s, 3), np.log(1 / 3), np.float32)
    for a, b in tramos_pasos(r["seccion"]):
        balon = torch.from_numpy(r["balon"][a:b]).float().to(device)[None]
        jug = torch.from_numpy(r["jug"][a:b].astype(np.float32)).to(device)[None]
        conv = torch.full((1, b - a), int(convencion), dtype=torch.long, device=device)
        logits, _ = red(balon, jug, conv)
        out[a:b] = torch.log_softmax(logits[0].float(), -1).cpu().numpy()
    return out


# TTA (E8): las 8 combinaciones de reflejo en x, reflejo en y y cambio de equipos, las mismas
# transformaciones que `aumentar` usa al entrenar.
TRANSFORMACIONES = [(fx, fy, sw) for fx in (0, 1) for fy in (0, 1) for sw in (0, 1)]


def transformar(r, fx, fy, sw):
    """Copia de los rasgos con reflejos (fx, fy) y cambio de equipos (sw), como en `aumentar`."""
    balon, jug = r["balon"].copy(), r["jug"].astype(np.float32).copy()
    for f, ij, ib in ((fx, JX, BX), (fy, JY, BY)):
        if f:
            jug[..., ij] *= -1
            balon[..., ib] *= -1
    if sw:
        jug[..., 1] *= -1
    return dict(r, balon=balon, jug=jug)


def predecir_tta(red, r, device="cpu", convencion=0, transformaciones=TRANSFORMACIONES):
    """log de la media de P(H, A, D) sobre las transformaciones; con sw se deshace H<->A."""
    lps = []
    for fx, fy, sw in transformaciones:
        lp = predecir(red, transformar(r, fx, fy, sw), device, convencion)
        lps.append(lp[:, [1, 0, 2]] if sw else lp)
    pila = np.stack(lps)
    m = pila.max(0)
    return (m + np.log(np.exp(pila - m).mean(0))).astype(np.float32)


def aumentar(balon, jug, etiq, real, gen):
    """Reflejos del campo y cambio de equipos (con H<->A). Todo en el lote, in place-safe."""
    balon, jug, etiq, real = balon.clone(), jug.clone(), etiq.clone(), real.clone()
    b = balon.shape[0]
    dev = balon.device
    fx = (torch.rand(b, generator=gen) < 0.5).to(dev)
    fy = (torch.rand(b, generator=gen) < 0.5).to(dev)
    sw = (torch.rand(b, generator=gen) < 0.5).to(dev)
    for f, ij, ib, ir in ((fx, JX, BX, 0), (fy, JY, BY, 1)):
        s = torch.where(f, -1.0, 1.0)
        jug[..., ij] *= s.view(b, 1, 1, 1).to(jug.dtype)
        balon[..., ib] *= s.view(b, 1, 1)
        real[..., ir] *= s.view(b, 1)
    s = torch.where(sw, -1.0, 1.0)
    jug[..., 1] *= s.view(b, 1, 1).to(jug.dtype)
    e = etiq
    swap = sw.view(b, 1) & (e >= 0) & (e < 2)
    etiq = torch.where(swap, 1 - e, e)
    return balon, jug, etiq, real
