"""Representación común de un partido: la misma estructura para el reto y para datos externos.

Todo predictor recibe un `Partido` y devuelve un array de códigos por frame (0=H, 1=A, 2=D, 3=X,
los mismos que evaluation/possession_metrics.py). Así cualquier método se evalúa igual en IDSSE
y en el ATD del reto, sin código específico para cada fuente.

Convenciones
------------
- Coordenadas normalizadas 0-1, como el ATD de Metrica: x a lo largo del campo, y a lo ancho
  con origen arriba. Para distancias en metros, multiplica por `campo` (largo, ancho).
- `jugadores[t, k]` es la posición del slot k en el frame t. NaN = no observado. Un slot es una
  pista, no una persona: en el ATD del reto hay 46 slots que aparecen, desaparecen y se
  intercambian. En datos externos limpios cada slot es un jugador, hasta que `degradar` los parte.
- `equipo_slot[k]`: 0 local, 1 visitante.
- `seccion[t]`: 0 fuera de juego evaluable, 1 primera parte, 2 segunda parte. El GT tiene X fuera.
- `gt` es opcional. En el reto solo existe el GT público (5 minutos), y solo sirve para comprobar
  que el pipeline está bien alineado, nunca para elegir ni ajustar. Ver RULES.md.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETRAS = np.array(["H", "A", "D", "X"])
FPS = 25


@dataclass
class Partido:
    nombre: str
    campo: tuple                      # (largo, ancho) en metros
    balon: np.ndarray                 # float32 [T, 2]
    jugadores: np.ndarray             # float32 [T, K, 2]
    equipo_slot: np.ndarray           # int8 [K]
    seccion: np.ndarray               # int8 [T]
    gt: Optional[np.ndarray] = None   # int8 [T], códigos H/A/D/X
    fps: int = FPS
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.balon)

    def validar(self):
        t = self.n_frames
        assert self.balon.shape == (t, 2), self.balon.shape
        assert self.jugadores.ndim == 3 and self.jugadores.shape[0] == t and self.jugadores.shape[2] == 2
        assert self.equipo_slot.shape == (self.jugadores.shape[1],)
        assert set(np.unique(self.equipo_slot)) <= {0, 1}
        assert self.seccion.shape == (t,)
        if self.gt is not None:
            assert self.gt.shape == (t,)
            assert set(np.unique(self.gt)) <= {H, A, D, X}
        return self

    def guardar(self, ruta):
        np.savez_compressed(
            ruta, nombre=self.nombre, campo=np.array(self.campo, dtype=np.float32),
            balon=self.balon, jugadores=self.jugadores, equipo_slot=self.equipo_slot,
            seccion=self.seccion, gt=self.gt if self.gt is not None else np.zeros(0, np.int8),
            fps=self.fps)

    @classmethod
    def cargar(cls, ruta):
        z = np.load(ruta, allow_pickle=False)
        gt = z["gt"] if len(z["gt"]) else None
        return cls(nombre=str(z["nombre"]), campo=tuple(float(v) for v in z["campo"]),
                   balon=z["balon"], jugadores=z["jugadores"], equipo_slot=z["equipo_slot"],
                   seccion=z["seccion"], gt=gt, fps=int(z["fps"])).validar()


def a_letras(codigos):
    return LETRAS[np.asarray(codigos)]


def escribir_csv(ruta, codigos):
    """CSV de entrega: frame,label. Mismo formato que valida evaluation/validate_submission.py."""
    letras = a_letras(codigos)
    with open(ruta, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(letras))
    return ruta
