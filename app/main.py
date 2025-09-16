from fastapi import FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "API rodando 🚀"}

# Contador de requisições
REQUEST_COUNT = Counter("requests_total", "Total de requisições ao modelo")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predict")
def predict():
    REQUEST_COUNT.inc()
    # lógica do modelo
    return {"prediction": "exemplo"}