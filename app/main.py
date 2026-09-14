from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path

from app.predictor import predecir

app = FastAPI(
    title="Hotel Bookings – Prediccion de cancelaciones",
    description="Modelo XGBoost que estima la probabilidad de que una reserva se cancele.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/app", StaticFiles(directory=BASE_DIR / "static", html=True), name="frontend")


class Reserva(BaseModel):
    """Una reserva de hotel.

    Todos los campos tienen valor por defecto para que se pueda llamar a
    /predict sin parametros y probar la API en un segundo. Cada campo que
    se pase por query string o por JSON sobrescribe su valor por defecto.
    """

    hotel: Literal["City Hotel", "Resort Hotel"] = "City Hotel"
    lead_time: int = Field(180, ge=0, description="Dias entre la reserva y la llegada")
    arrival_date_year: int = 2017
    arrival_date_month: Literal[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ] = "August"
    arrival_date_week_number: int = 34
    arrival_date_day_of_month: int = 20
    stays_in_weekend_nights: int = Field(2, ge=0)
    stays_in_week_nights: int = Field(5, ge=0)
    adults: int = Field(2, ge=0)
    children: float = Field(1, ge=0)
    babies: int = Field(0, ge=0)
    meal: str = "BB"
    country: str = "ESP"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = Field(0, ge=0, le=1)
    previous_cancellations: int = Field(0, ge=0)
    previous_bookings_not_canceled: int = Field(0, ge=0)
    reserved_room_type: str = "A"
    assigned_room_type: str = "A"
    booking_changes: int = Field(0, ge=0)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = "No Deposit"
    agent: float = 240
    days_in_waiting_list: int = Field(0, ge=0)
    customer_type: str = "Transient"
    adr: float = Field(140.0, description="Average Daily Rate: precio medio por noche")
    required_car_parking_spaces: int = Field(1, ge=0)
    total_of_special_requests: int = Field(2, ge=0)


@app.get("/")
def landing():
    """Pagina de inicio: explica como usar el resto de endpoints."""
    return {
        "api": "Prediccion de cancelaciones de reservas de hotel",
        "modelo": "XGBoost (ROC-AUC 0.95 en test)",
        "endpoints": {
            "GET /": "esta pagina",
            "GET /health": "comprueba que el servicio esta vivo",
            "GET /predict": "prediccion pasando los datos por query string",
            "POST /predict": "prediccion pasando los datos en un JSON",
            "GET /docs": "documentacion interactiva (Swagger) para probar la API desde el navegador",
        },
        "ejemplo_get": "/predict?lead_time=350&deposit_type=Non%20Refund&required_car_parking_spaces=0",
        "ejemplo_python": (
            "import requests; "
            "requests.get('https://TU-URL/predict', params={'lead_time': 350}).json()"
        ),
        "campos_admitidos": list(Reserva.model_fields.keys()),
        "nota": "Todos los campos son opcionales: los que no se envien usan el valor por defecto del ejemplo.",
    }


@app.get("/health")
def health():
    """Endpoint de vida: lo usan Render y Kubernetes para saber si reiniciar el contenedor."""
    return {"status": "ok"}


@app.get("/predict")
def predict_get(reserva: Reserva = Depends()):
    """Prediccion via GET: los campos del modelo se leen de la query string."""
    return _predecir_o_error(reserva)


@app.post("/predict")
def predict_post(reserva: Reserva):
    """Prediccion via POST: los campos del modelo llegan en el cuerpo JSON."""
    return _predecir_o_error(reserva)


def _predecir_o_error(reserva: Reserva):
    """Envuelve la prediccion para devolver un 400 legible en vez de un 500 opaco."""
    try:
        resultado = predecir(reserva.model_dump())
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"No se ha podido calcular la prediccion con esos datos: {error}",
        )
    return {"entrada": reserva.model_dump(), **resultado}


# --- TERCER ENDPOINT: descomentar y redesplegar en directo -------------------
# @app.get("/modelo")
# def info_modelo():
#     """Ficha tecnica del modelo que hay detras de la API."""
#     from app.predictor import modelo, pipeline
#
#     return {
#         "algoritmo": type(modelo).__name__,
#         "n_features": modelo.n_features_in_,
#         "pasos_del_pipeline": [nombre for nombre, _ in pipeline.steps],
#         "metricas_en_test": {"roc_auc": 0.95},
#     }
