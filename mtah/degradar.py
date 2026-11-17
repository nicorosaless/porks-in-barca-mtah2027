"""Degrada un partido limpio (p. ej. IDSSE) hasta parecerse al ATD automático del reto.

Los datos externos vienen de tracking corregido: balón en todos los frames, un slot por jugador,
sin huecos. Validar ahí sobreestima cualquier método. Esta función mete los defectos que DATA.md
describe en el ATD, calibrados con estadísticas agregadas del ATD del reto
(`python scripts/calidad_tracking.py`):

    balón ausente        40.7 % de frames evaluables, en bloques largos (mediana ~270 frames)
    fragmentación        pistas de mediana ~900 frames; 46 slots para 22 jugadores
    jugadores visibles   20.5 de media

Y quita un artefacto de DFL que no existe en el ATD: con el balón muerto, IDSSE congela su
posición (desplazamiento exactamente 0 en el 100 % de los frames muertos; en el ATD, 0.1 % de los
frames con balón). Sin quitarlo, un modelo aprende "balón quieto = muerto" y la validación sale
inflada (iteración 2: spellF1 0.65 en IDSSE). Cada tramo congelado se convierte en balón ausente
(con más probabilidad si está fuera del campo, donde la cámara táctica apenas lo ve) o en un
movimiento lento sintético. Los parámetros igualan el histograma de velocidad del balón y la
fracción fuera del campo del ATD (`python scripts/hibrido/calibrar_balon.py`).

Lo que sigue SIN modelar en v2/v3: dónde faltan los huecos aleatorios (DATA.md dice que más en
duelos y áreas congestionadas); aquí se colocan al azar.

Escenario v4 (E5, 2026-11-13). Todo viene apagado por defecto, así que v2/v3 no cambian ni un bit:
- `aleatorizar`: los parámetros calibrados con agregados NO publicados del ATD (huecos, fragmentos,
  ruido) se sortean por semilla dentro de rangos amplios que contienen el valor calibrado. Fijas
  solo quedan las cifras publicadas en DATA.md (40,7 % sin balón y 46 slots).
- Defectos que describe DATA.md: jugadores duplicados (a veces hay más de 22 visibles), huecos
  de balón más probables con el balón rápido o aéreo, balón falso (el tracker se engancha a otra
  cosa) y equipo erróneo al recuperar una pista. Tasas fijadas a priori, sin datos del reto.

Todo es determinista dado `semilla`.
"""
from dataclasses import dataclass, replace

import numpy as np

from mtah.partido import Partido


@dataclass(frozen=True)
class Degradacion:
    balon_ausente: float = 0.407         # fracción objetivo de frames evaluables sin balón
    # Calibrado el 2026-11-04 contra los cuantiles agregados del ATD (scripts/calidad_tracking.py,
    # IDSSE J03WMX). Huecos de balón resultantes q50/q75/q90 ~ 200/500/930 frames frente a
    # 268/527/845 en el ATD; jugadores visibles 21.2 frente a 20.5. Solo estadísticas del input,
    # sin etiquetas del reto.
    hueco_balon_mediana: float = 200.0   # frames; lognormal (los huecos que se solapan se funden)
    hueco_balon_sigma: float = 0.9       # sigma del log
    fragmento_mediana: float = 900.0     # frames por pista antes de cortarse
    fragmento_sigma: float = 1.5
    hueco_jugador_media: float = 90.0    # frames que un jugador desaparece entre fragmentos
    slots_por_equipo: int = 23           # el ATD del reto tiene 24 y 22
    ruido_m: float = 0.15                # ruido gaussiano de posición, metros
    # Balón congelado de DFL (2026-11-07). False reproduce la degradación original.
    quitar_congelado: bool = True
    congelado_ausente: float = 0.7       # prob. de que un tramo congelado dentro del campo falte
    congelado_fuera_ausente: float = 0.9  # ídem si el balón está fuera del campo
    congelado_vel_mediana: float = 2.5   # m/s del movimiento sintético (lognormal)
    # (0.7, 0.9, 2.5): L1 0.077 frente al ATD en visibilidad, fuera y velocidad; original 1.013
    ruido_balon_m: float = 0.03          # jitter del balón observado, metros
    hueco_peso_cerca: float = 0.0
    hueco_peso_area: float = 0.0
    hueco_peso_fuera: float = 0.0
    hueco_minimo_largo: int = 1
    hueco_corto_prob: float = 0.0
    # v4
    aleatorizar: bool = False
    hueco_peso_rapido: float = 0.0       # peso extra de hueco con el balón limpio a > 15 m/s
    duplicados_por_min: float = 0.0      # copias de un jugador en un slot extra
    balon_falso_por_min: float = 0.0     # tramos con el balón observado en otro sitio
    equipo_erroneo: float = 0.0          # prob. de que un fragmento caiga en el otro equipo
    semilla: int = 0


# Rangos de `aleatorizar`: (mínimo, máximo, logarítmico). Contienen el valor calibrado.
RANGOS = {
    "hueco_balon_mediana": (120.0, 400.0, True),
    "hueco_balon_sigma": (0.6, 1.2, False),
    "fragmento_mediana": (300.0, 2000.0, True),
    "fragmento_sigma": (1.0, 1.8, False),
    "hueco_jugador_media": (40.0, 200.0, True),
    "ruido_m": (0.08, 0.35, False),
}


def aleatorizada(cfg: Degradacion) -> Degradacion:
    """Con `aleatorizar`, sortea los parámetros de RANGOS con un generador propio de la semilla."""
    if not cfg.aleatorizar:
        return cfg
    rng = np.random.default_rng([cfg.semilla, 2027])
    valores = {}
    for k, (lo, hi, log) in RANGOS.items():
        u = rng.random()
        valores[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if log else lo + u * (hi - lo)
    return replace(cfg, **valores)


def degradar(p: Partido, cfg: Degradacion = Degradacion()) -> Partido:
    cfg = aleatorizada(cfg)
    rng = np.random.default_rng(cfg.semilla)
    balon = _quitar_congelado(p, cfg, rng) if cfg.quitar_congelado else p.balon.copy()
    balon = _quitar_balon(p, balon, cfg, rng)
    jug, equipo_slot = _fragmentar(p, cfg, rng)
    if cfg.duplicados_por_min > 0:
        jug, equipo_slot = _duplicar(p, jug, equipo_slot, cfg, rng)
    if cfg.balon_falso_por_min > 0:
        balon = _balon_falso(p, balon, jug, cfg, rng)
    if cfg.ruido_m > 0:
        escala = np.array([cfg.ruido_m / p.campo[0], cfg.ruido_m / p.campo[1]], np.float32)
        jug = jug + rng.normal(0, 1, jug.shape).astype(np.float32) * escala
    if cfg.quitar_congelado and cfg.ruido_balon_m > 0:
        escala = np.array([cfg.ruido_balon_m / p.campo[0], cfg.ruido_balon_m / p.campo[1]],
                          np.float32)
        balon = balon + rng.normal(0, 1, balon.shape).astype(np.float32) * escala
    q = replace(p, nombre="{}_deg{}".format(p.nombre, cfg.semilla), balon=balon,
                jugadores=jug, equipo_slot=equipo_slot, extra=dict(p.extra, degradacion=cfg))
    return q.validar()


def tramos_congelados(balon, minimo=5):
    """[(ini, fin)] de tramos con el balón exactamente quieto durante >= `minimo` frames."""
    quieto = np.zeros(len(balon), bool)
    d = np.diff(balon, axis=0)
    quieto[1:] = (d[:, 0] == 0) & (d[:, 1] == 0)
    # el primer frame del tramo también es congelado
    quieto[:-1] |= quieto[1:]
    e = np.diff(np.concatenate(([0], quieto.astype(np.int8), [0])))
    return [(a, b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a >= minimo]


def _quitar_congelado(p, cfg, rng):
    balon = p.balon.copy()
    largo, ancho = p.campo
    for a, b in tramos_congelados(balon):
        x, y = balon[a]
        fuera = x < 0 or x > 1 or y < 0 or y > 1
        if rng.random() < (cfg.congelado_fuera_ausente if fuera else cfg.congelado_ausente):
            balon[a:b] = np.nan
            continue
        # paseo por tramos de velocidad constante (balón conducido, colocado, pasado de mano)
        n = b - a
        pasos = np.zeros((n, 2), np.float32)
        i = 0
        while i < n:
            dur = int(max(5, rng.exponential(p.fps)))
            v = rng.lognormal(np.log(cfg.congelado_vel_mediana), 0.7) / p.fps
            ang = rng.uniform(0, 2 * np.pi)
            pasos[i:i + dur] = [v * np.cos(ang) / largo, v * np.sin(ang) / ancho]
            i += dur
        tray = np.array([x, y], np.float32) + np.cumsum(pasos, axis=0)
        balon[a:b] = np.clip(tray, -0.05, 1.05)
    return balon


def _quitar_balon(p, balon, cfg, rng):
    evaluable = p.seccion > 0
    objetivo = cfg.balon_ausente * evaluable.sum()
    falta = np.isnan(balon[:, 0]) & evaluable
    candidatos = np.flatnonzero(evaluable)
    pesos = _pesos_hueco(p, balon, cfg, candidatos)
    while falta.sum() < objetivo:
        if cfg.hueco_corto_prob > 0 and rng.random() < cfg.hueco_corto_prob:
            largo = int(rng.integers(2, 25))
        else:
            largo = int(max(cfg.hueco_minimo_largo,
                            rng.lognormal(np.log(cfg.hueco_balon_mediana),
                                          cfg.hueco_balon_sigma)))
        ini = int(rng.choice(candidatos, p=pesos))
        fin = min(ini + largo, len(balon))
        # El hueco no cruza de una parte a otra
        fin = ini + int(np.argmax(p.seccion[ini:fin] != p.seccion[ini])) if (p.seccion[ini:fin] != p.seccion[ini]).any() else fin
        falta[ini:fin] = True
    balon[falta] = np.nan
    return balon


def _pesos_hueco(p, balon, cfg, candidatos):
    w = np.ones(len(candidatos), float)
    obs = ~np.isnan(balon[candidatos, 0])
    if obs.any():
        w[~obs] = 0.0
    x, y = balon[candidatos, 0], balon[candidatos, 1]
    fuera = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    area = ((x < 0.17) | (x > 0.83)) & (y > 0.2) & (y < 0.8)
    if cfg.hueco_peso_fuera:
        w += cfg.hueco_peso_fuera * fuera
    if cfg.hueco_peso_area:
        w += cfg.hueco_peso_area * area
    if cfg.hueco_peso_rapido:
        v = np.zeros(len(p.balon))
        with np.errstate(all="ignore"):
            v[1:] = np.linalg.norm(np.diff(p.balon, axis=0) * np.array(p.campo), axis=1) * p.fps
        w += cfg.hueco_peso_rapido * (np.nan_to_num(v[candidatos]) > 15.0)
    if cfg.hueco_peso_cerca and p.jugadores.shape[1] > 0:
        bm = balon[candidatos] * np.array(p.campo, np.float32)
        jm = p.jugadores[candidatos] * np.array(p.campo, np.float32)
        d = np.sqrt(((jm - bm[:, None, :]) ** 2).sum(-1))
        with np.errstate(all="ignore"):
            dmin = np.nanmin(d, axis=1)
        w += cfg.hueco_peso_cerca * (dmin < 2.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() == 0:
        w[:] = 1.0
    return w / w.sum()


def _fragmentar(p, cfg, rng):
    """Corta cada jugador en fragmentos y los reasigna a slots anónimos por equipo.

    Un slot se reutiliza cuando queda libre, como hace un tracker que pierde y recupera a gente:
    el mismo slot puede pasar de un jugador a otro, y un jugador cambia de slot al reaparecer.
    """
    t, k, _ = p.jugadores.shape
    fragmentos = []  # (inicio, fin, jugador_original)
    for j in range(k):
        vis = ~np.isnan(p.jugadores[:, j, 0])
        if not vis.any():
            continue
        d = np.diff(np.concatenate(([0], vis.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            pos = s
            while pos < e:
                largo = int(max(25, rng.lognormal(np.log(cfg.fragmento_mediana), cfg.fragmento_sigma)))
                fin = min(pos + largo, e)
                fragmentos.append((pos, fin, j))
                pos = fin + int(rng.exponential(cfg.hueco_jugador_media))
    fragmentos.sort()

    libres_desde = {0: [], 1: []}  # por equipo: frame desde el que cada slot está libre
    asignacion = []
    for s, e, j in fragmentos:
        eq = int(p.equipo_slot[j])
        if cfg.equipo_erroneo > 0 and rng.random() < cfg.equipo_erroneo:
            eq = 1 - eq
        slots = libres_desde[eq]
        # Un fragmento nuevo cae en un slot libre cualquiera o en uno nuevo (hasta el tope), al
        # azar: así el número de slots crece hacia el del ATD (24/22) en vez de quedarse en 11.
        opciones = [i for i, f in enumerate(slots) if f <= s]
        if len(slots) < cfg.slots_por_equipo or not opciones:
            opciones.append(len(slots))
        libre = int(rng.choice(opciones))
        if libre == len(slots):
            slots.append(e)
        else:
            slots[libre] = e
        asignacion.append((s, e, j, eq, libre))

    n0 = len(libres_desde[0])
    equipo_slot = np.array([0] * n0 + [1] * len(libres_desde[1]), np.int8)
    jug = np.full((t, len(equipo_slot), 2), np.nan, np.float32)
    for s, e, j, eq, libre in asignacion:
        destino = libre if eq == 0 else n0 + libre
        jug[s:e, destino] = p.jugadores[s:e, j]
    return jug, equipo_slot


def _n_eventos(p, por_min, rng):
    minutos = (p.seccion > 0).sum() / p.fps / 60.0
    return int(rng.poisson(por_min * minutos))


def _duplicar(p, jug, equipo_slot, cfg, rng):
    """El tracker duplica a alguien: una copia desplazada en un slot libre del mismo equipo (o uno nuevo)."""
    t, k, _ = jug.shape
    evaluables = np.flatnonzero(p.seccion > 0)
    extra = {0: [], 1: []}  # por equipo: [(columna nueva, libre desde)]
    nuevas = []
    for _ in range(_n_eventos(p, cfg.duplicados_por_min, rng)):
        f = int(rng.choice(evaluables))
        vis = np.flatnonzero(~np.isnan(jug[f, :, 0]))
        if not len(vis):
            continue
        j = int(rng.choice(vis))
        dur = int(np.clip(rng.lognormal(np.log(50), 0.7), 5, 500))
        fin = min(f + dur, t)
        off = rng.normal(0, 0.7, 2) / np.array(p.campo)
        eq = int(equipo_slot[j])
        huecos = np.flatnonzero((equipo_slot[:jug.shape[1]] == eq)
                                & np.isnan(jug[f:fin, :len(equipo_slot), 0]).all(0))
        libres = [i for i, (_, desde) in enumerate(extra[eq]) if desde <= f]
        if len(huecos):
            col = int(rng.choice(huecos))
        elif libres:
            col = extra[eq][libres[0]][0]
            extra[eq][libres[0]] = (col, fin)
        else:
            col = k + len(nuevas)
            nuevas.append(eq)
            extra[eq].append((col, fin))
        nuevas_cols = len(nuevas)
        if jug.shape[1] < k + nuevas_cols:
            jug = np.concatenate([jug, np.full((t, k + nuevas_cols - jug.shape[1], 2), np.nan, np.float32)], 1)
        jug[f:fin, col] = jug[f:fin, j] + off.astype(np.float32)
    return jug, np.concatenate([equipo_slot, np.array(nuevas, np.int8)])


def _balon_falso(p, balon, jug, cfg, rng):
    """El tracker se engancha a otra cosa: el balón salta a un jugador o a un punto cualquiera."""
    balon = balon.copy()
    t = len(balon)
    evaluables = np.flatnonzero(p.seccion > 0)
    for _ in range(_n_eventos(p, cfg.balon_falso_por_min, rng)):
        f = int(rng.choice(evaluables))
        dur = int(np.clip(rng.lognormal(np.log(12), 0.6), 3, 75))
        fin = min(f + dur, t)
        vis = np.flatnonzero(~np.isnan(jug[f, :, 0]))
        if len(vis) and rng.random() < 0.5:
            destino = jug[f:fin, int(rng.choice(vis))].copy()
        else:
            destino = np.tile(rng.uniform(0.02, 0.98, 2).astype(np.float32), (fin - f, 1))
        balon[f:fin] = destino
    return balon
