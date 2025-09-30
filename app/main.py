from fastapi import FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.recruiter import find_top_applicants_with_filters
import os
import yaml
import sys
from pathlib import Path
from pydantic import BaseModel


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Import or define the missing variables
from src.recruiter import RecruiterBot  # Assuming Bot is defined in src.bot
from src.indexer import FAISSIndexer  # Assuming FaissIndexer is defined in src.indexer
from src.embedding_manager import EmbeddingManager  # Assuming EmbeddingManager is defined in src.embeddings

# Initialize the missing variables
with open(os.path.join( "src", "config", "index_config.yaml")) as f:
    index_cfg = yaml.safe_load(f)
indexer = FAISSIndexer( index_cfg)
emb_mgr = EmbeddingManager('src/models_config.yaml')
bot = RecruiterBot(emb_mgr, indexer)

app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "API rodando 🚀"}

# Contador de requisições
REQUEST_COUNT = Counter("requests_total", "Total de requisições ao modelo")

@app.get("/metrics")
def metrics():
    print("Metrics endpoint called")
    print(generate_latest().decode('utf-8'))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class PredictRequest(BaseModel):
    job_description: str
    top_n: int = 10
    search_k: int = 100

@app.post("/predict")
def predict_post(req: PredictRequest):
    REQUEST_COUNT.inc()
    result = find_top_applicants_with_filters(
        job_description=req.job_description,
        faiss_indexer=indexer,
        emb_mgr=emb_mgr,
        filters=bot.filters if getattr(bot, "filters", None) else None,
        top_n=req.top_n,
        search_k=req.search_k,
    )
    return result

