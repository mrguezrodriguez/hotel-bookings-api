"""
Reempaqueta full_pipeline.pkl para que deje de depender de __main__.

El pipeline se guardó desde el notebook, así que dentro lleva escrito
"__main__.ColumnDropper". Al cargarlo desde la API eso falla, porque ahí
__main__ es uvicorn y no tiene esas clases.

Este script se ejecuta a mano, una sola vez:
  1. importa las clases desde app.transformers_hotel al espacio de __main__
     (de ahí el import *), con lo que el pickle viejo ya puede cargarse
  2. vuelve a guardar el objeto. Como las clases ahora están en un módulo
     propio, el pickle nuevo apunta a "app.transformers_hotel.ColumnDropper"

Uso, desde la raíz del repo:
    python repack_model.py
"""

import joblib
from pathlib import Path

from app.transformers_hotel import *  # noqa: F401,F403

ORIGEN = Path("models_origen/full_pipeline.pkl")
DESTINO = Path("app/models/full_pipeline.pkl")


def main():
    print(f"Cargando {ORIGEN} ...")
    pipeline = joblib.load(ORIGEN)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, DESTINO)
    print(f"Guardado {DESTINO}")

    # Comprobacion - recarga en proceso limpio simulado
    recargado = joblib.load(DESTINO)
    print("Pasos del pipeline:", [nombre for nombre, _ in recargado.steps])
    print("Modulo de la primera clase:", type(recargado.steps[0][1]).__module__)


if __name__ == "__main__":
    main()
