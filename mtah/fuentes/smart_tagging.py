"""Smart Tagging del reto (SportsCode XML) -> lista de `Instancia`.

Misma lectura que starter/load_smart_tagging.py, con la librería estándar: <start> y <end> son
segundos desde el primer frame del vídeo táctico, así que round(s * 25) es el frame (DATA.md,
"Time base -- verified"). Intervalos semiabiertos: `fin_frame` es el primer frame después.

`Team` se pasa a H o A con la convención de mtah/fuentes/reto.py: el primer `Team` del XML de
metadatos del ATD es el local. El cruce es por nombre (sin mayúsculas ni espacios de más); un
nombre que no casa es un error, no una suposición. `N/A` o sin `Team` -> SIN_EQUIPO.

Es un dato del reto: este módulo nunca imprime instancias. `agregados` y el `__main__` dan solo
cuentas por código, el número de `N/A` y el reparto H/A.
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import NamedTuple

from mtah.partido import A, FPS, H
from mtah.rutas import RAIZ

SIN_EQUIPO = -1
_PARTES = {"1st half": 1, "2nd half": 2}


class Instancia(NamedTuple):
    inicio_frame: int   # frame de vídeo, incluido
    fin_frame: int      # frame de vídeo, excluido
    code: str           # en mayúsculas: "SET PIECES", "BUILD UP"...
    team: int           # H, A o SIN_EQUIPO
    type: str           # grupo `Type` tal cual ("Throw-in"), "" si no lo tiene
    half: int           # 1, 2 o 0 si no lo dice


def _norm(s):
    return " ".join((s or "").split()).lower()


def equipos_atd(meta_path):
    """[local, visitante]: nombres de los `Team` del XML de metadatos del ATD, en orden."""
    root = ET.parse(meta_path).getroot()
    nombres = [t.findtext("Name") for t in root.iter("Team")]
    if len(nombres) != 2:
        raise ValueError("{}: se esperaban 2 equipos".format(meta_path))
    return nombres


def parsear(ruta_xml, equipos, fps=FPS):
    """Instancias en orden de fichero. `equipos` = [local, visitante] (ver `equipos_atd`)."""
    lado = {_norm(equipos[0]): H, _norm(equipos[1]): A}
    out = []
    for inst in ET.parse(ruta_xml).iter("instance"):
        # un grupo repetido: gana el último, como en el starter (Team, Half y Type son únicos)
        tags = {_norm(l.findtext("group")): (l.findtext("text") or "").strip()
                for l in inst.findall("label")}
        team = _norm(tags.get("team"))
        if team in ("", "n/a"):
            team = SIN_EQUIPO
        elif team in lado:
            team = lado[team]
        else:
            raise ValueError("Team {!r} no casa con los equipos del ATD {}".format(
                tags.get("team"), list(equipos)))
        out.append(Instancia(
            int(round(float(inst.findtext("start")) * fps)),
            int(round(float(inst.findtext("end")) * fps)),
            " ".join((inst.findtext("code") or "").split()).upper(),
            team,
            tags.get("type", ""),
            _PARTES.get(_norm(tags.get("half")), 0),
        ))
    return out


def cargar(game=None, data_dir=None):
    """Instancias del Smart Tagging del partido del reto, con H/A según el ATD táctico."""
    sys.path.insert(0, RAIZ)
    from games import GAME, require  # noqa: E402

    st, meta = require(["smart_tagging", "atd_meta"], game or GAME, data_dir)
    return parsear(st, equipos_atd(meta))


def agregados(instancias):
    """Solo cuentas: por código, `N/A` y reparto H/A. Lo único que se puede enseñar."""
    return {
        "por_codigo": dict(sorted(Counter(i.code for i in instancias).items())),
        "n_na": sum(i.team == SIN_EQUIPO for i in instancias),
        "H": sum(i.team == H for i in instancias),
        "A": sum(i.team == A for i in instancias),
    }


if __name__ == "__main__":
    ag = agregados(cargar())
    print("instancias: {}".format(sum(ag["por_codigo"].values())))
    for code, n in sorted(ag["por_codigo"].items(), key=lambda kv: -kv[1]):
        print("  {:<32} {:>4}".format(code, n))
    print("Team: H {}  A {}  N/A {}".format(ag["H"], ag["A"], ag["n_na"]))
