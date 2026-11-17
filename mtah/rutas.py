"""Rutas del proyecto. Nada de aquí apunta a un fichero concreto del reto: eso es cosa de games.py."""
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def external_dir():
    """Datos de otros partidos (IDSSE, etc.). Por defecto data/external, que git ignora.

    Con varios worktrees, apunta MTAH_EXTERNAL_DIR a una sola carpeta para no duplicar gigas.
    """
    return os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(RAIZ, "data", "external")


def cache_dir():
    """Conversiones intermedias (npz) de datos externos. Regenerables; ignoradas por git."""
    return os.path.join(external_dir(), "cache")
