from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.recruiter import find_top_applicants_with_filters
import os
import yaml
import sys
from pathlib import Path
from pydantic import BaseModel
import time


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Import or define the missing variables
from src.recruiter import RecruiterBot  # Assuming Bot is defined in src.bot
from src.indexer import FAISSIndexer  # Assuming FaissIndexer is defined in src.indexer
from src.embedding_manager import EmbeddingManager  # Assuming EmbeddingManager is defined in src.embeddings

# Import métricas melhoradas
from src.metrics import (
    REQUESTS_TOTAL, REQUEST_DURATION, CANDIDATES_FOUND, CANDIDATE_SCORES,
    SEARCHES_BY_AREA, FAISS_SEARCH_DURATION, ACTIVE_CANDIDATES_COUNT,
    COMPONENT_HEALTH, APPLICATION_INFO, update_system_metrics,
    classify_job_area, extract_experience_level, track_endpoint_metrics
)

# Initialize the missing variables
with open(os.path.join( "src", "config", "index_config.yaml")) as f:
    index_cfg = yaml.safe_load(f)
indexer = FAISSIndexer( index_cfg)
emb_mgr = EmbeddingManager('src/models_config.yaml')
bot = RecruiterBot(emb_mgr, indexer)

app = FastAPI(title="Job Matching API", version="1.0.0")

# Configurar métricas de saúde dos componentes na inicialização
COMPONENT_HEALTH.labels(component="faiss").set(1)
COMPONENT_HEALTH.labels(component="embeddings").set(1)
COMPONENT_HEALTH.labels(component="database").set(1)

@app.get("/")
def home():
    return {"mensagem": "API rodando 🚀", "version": "1.0.0", "status": "healthy"}

@app.get("/health")
def health_check():
    """Endpoint de health check para monitoramento"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "faiss": "healthy",
            "embeddings": "healthy", 
            "database": "healthy"
        }
    }

@app.get("/metrics")
def metrics():
    """Endpoint de métricas para Prometheus/Grafana com métricas melhoradas"""
    print("Metrics endpoint called")
    
    # Atualizar métricas de sistema antes de retornar
    update_system_metrics()
    
    # Log das métricas principais para debug
    metrics_output = generate_latest().decode('utf-8')
    print("=== MÉTRICAS PRINCIPAIS ===")
    for line in metrics_output.split('\n'):
        if 'job_matching' in line and not line.startswith('#'):
            print(line)
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/summary")
def metrics_summary():
    """Endpoint para visualizar resumo das métricas de forma amigável"""
    metrics_text = generate_latest().decode('utf-8')
    
    summary = {
        "timestamp": time.time(),
        "application": {
            "status": "running",
            "version": "1.0.0"
        },
        "requests": {},
        "performance": {},
        "business": {}
    }
    
    # Parsear métricas básicas do output
    for line in metrics_text.split('\n'):
        if line and not line.startswith('#'):
            if 'job_matching_requests_total' in line:
                summary["requests"]["total_requests"] = line.split()[-1]
            elif 'job_matching_candidates_found' in line:
                summary["business"]["candidates_found"] = line.split()[-1]
            elif 'job_matching_request_duration' in line:
                summary["performance"]["avg_duration_seconds"] = line.split()[-1]
    
    return summary


class PredictRequest(BaseModel):
    job_description: str
    top_n: int = 10
    search_k: int = 100

@app.post("/predict")
@track_endpoint_metrics("predict")
def predict_post(req: PredictRequest):
    """Endpoint para predição de candidatos com métricas melhoradas"""
    start_time = time.time()
    
    # Classificar área e nível de experiência para métricas
    job_area = classify_job_area(req.job_description)
    experience_level = extract_experience_level(req.job_description)
    
    # Incrementar contador por área
    SEARCHES_BY_AREA.labels(
        job_area=job_area,
        experience_level=experience_level
    ).inc()
    
    # Buscar candidatos
    result = find_top_applicants_with_filters(
        job_description=req.job_description,
        faiss_indexer=indexer,
        emb_mgr=emb_mgr,
        filters=bot.filters if getattr(bot, "filters", None) else None,
        top_n=req.top_n,
        search_k=req.search_k,
    )
    
    # Registrar métricas de resultado
    if result:
        CANDIDATES_FOUND.labels(search_type="predict").observe(len(result))
        
        # Registrar distribuição de scores
        for candidate in result:
            if 'score' in candidate:
                score = candidate['score']
                CANDIDATE_SCORES.labels(score_range="all").observe(score)
    
    # Registrar tempo total de processamento
    total_duration = time.time() - start_time
    REQUEST_DURATION.labels(endpoint="predict", method="POST").observe(total_duration)
    
    return result

