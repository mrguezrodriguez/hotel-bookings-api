"""
Prueba la API con requests, exactamente como la probara quien la corrija.

Uso:
    python tests/test_api.py                      # contra localhost
    python tests/test_api.py https://mi-api.onrender.com
"""

import sys

import requests

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def comprobar(nombre, condicion):
    print("\nTODO CORRECTO" if ok else "\nHay comprobaciones que fallan")
    if not ok:
        sys.exit(1)


def main():
    ok = True

    r = requests.get(f"{BASE}/")
    ok &= comprobar("Landing responde 200", r.status_code == 200)
    ok &= comprobar("Landing explica los endpoints", "endpoints" in r.json())

    r = requests.get(f"{BASE}/predict")
    ok &= comprobar("GET /predict sin parametros", r.status_code == 200)
    print("   ->", r.json()["etiqueta"], r.json()["probabilidad_cancelacion"])

    r = requests.get(
        f"{BASE}/predict",
        params={
            "lead_time": 350,
            "deposit_type": "Non Refund",
            "required_car_parking_spaces": 0,
            "previous_cancellations": 2,
        },
    )
    ok &= comprobar("GET /predict con parametros", r.status_code == 200)
    print("   ->", r.json()["etiqueta"], r.json()["probabilidad_cancelacion"])

    r = requests.post(f"{BASE}/predict", json={"lead_time": 10, "market_segment": "Direct"})
    ok &= comprobar("POST /predict con JSON", r.status_code == 200)

    r = requests.get(f"{BASE}/predict", params={"lead_time": "ochenta"})
    ok &= comprobar("Dato invalido -> 422 y no 500", r.status_code == 422)

    r = requests.get(f"{BASE}/predict", params={"arrival_date_month": "Agosto"})
    ok &= comprobar("Mes en castellano -> 422", r.status_code == 422)

    print("\nTODO CORRECTO" if ok else "\nHay comprobaciones que fallan")


if __name__ == "__main__":
    main()
