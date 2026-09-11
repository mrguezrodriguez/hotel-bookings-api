# Imagen base "slim": Python ya instalado, sin las 400 MB extra de la completa
FROM python:3.11-slim

# Buenas practicas de Python dentro de contenedores:
# - no generar archivos .pyc
# - no bufferizar la salida, para que los logs salgan en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Las dependencias se copian ANTES que el codigo: Docker cachea cada capa,
# asi que si solo cambia el codigo no reinstala 300 MB de librerias.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Render y Kubernetes inyectan el puerto por variable de entorno.
ENV PORT=8000
EXPOSE 8000

# Forma "shell" (sin corchetes) a proposito: asi se expande $PORT.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
