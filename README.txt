# ---------------- Levantar la API -------------- #
uvicorn app.main:app --host 0.0.0.0 --port 8000
# ----------------------------------------------- #

# ---------- Construir la imagen docker --------- #
docker build --no-cache -t mlops-api-demo .
# ----------------------------------------------- #

# ----------- Ejecutar la imagen docker --------- #
docker run -p 8000:8000 mlops-api-demo
# ----------------------------------------------- #