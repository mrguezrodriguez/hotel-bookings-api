"""
Carga el pipeline y el modelo una sola vez y expone la función predecir().

La carga ocurre al importar el módulo, o sea al arrancar la API, y no dentro de
predecir(): leer los .pkl tarda alrededor de un segundo, así que si estuviera
dentro, ese segundo se pagaría en cada petición.
"""

from pathlib import Path

import joblib
import pandas as pd

# El pipeline devuelve estas tres columnas como numeros, pero el XGBoost se
# entreno con ellas como 'category'. Si no se convierten, xgboost lanza
# "The data type doesn't match the one used in the training dataset".
COLS_BIN = ["lead_time_bin", "adr_bin", "total_nights_bin"]
TIPO_BIN = pd.CategoricalDtype(categories=[0, 1, 2, 3])

RUTA_MODELOS = Path(__file__).parent / "models"

pipeline = joblib.load(RUTA_MODELOS / "full_pipeline.pkl")
modelo = joblib.load(RUTA_MODELOS / "xgboost_final.pkl")


def predecir(datos: dict) -> dict:
    """Recibe una reserva como diccionario y devuelve prediccion + probabilidad."""
    reserva = pd.DataFrame([datos])

    preparada = pipeline.transform(reserva)
    for col in COLS_BIN:
        preparada[col] = preparada[col].astype("int64").astype(TIPO_BIN)

    prediccion = int(modelo.predict(preparada)[0])
    probabilidad = float(modelo.predict_proba(preparada)[0, 1])

    return {
        "prediccion": prediccion,
        "etiqueta": "Se cancelara" if prediccion == 1 else "No se cancelara",
        "probabilidad_cancelacion": round(probabilidad, 4),
        "probabilidad_no_cancelacion": round(1 - probabilidad, 4),
    }
