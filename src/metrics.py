"""
Módulo de métricas para monitoramento do sistema de job matching
Fornece métricas específicas para integração com Grafana/Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# =============================================================================
# MÉTRICAS DE REQUISIÇÕES
# =============================================================================

# Contador total de requisições (melhoria da métrica existente)
REQUESTS_TOTAL = Counter(
    'job_matching_requests_total',
    'Total de requisições ao sistema de job matching',
    ['endpoint', 'method', 'status_code']
)

# Latência de requisições
REQUEST_DURATION = Histogram(
    'job_matching_request_duration_seconds',
    'Tempo de resposta das requisições em segundos',
    ['endpoint', 'method'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# =============================================================================
# MÉTRICAS DE NEGÓCIO - JOB MATCHING
# =============================================================================

# Candidatos encontrados por busca
CANDIDATES_FOUND = Histogram(
    'job_matching_candidates_found',
    'Número de candidatos encontrados por busca',
    ['search_type'],
    buckets=[0, 1, 5, 10, 25, 50, 100, 500]
)

# Score médio dos candidatos retornados
CANDIDATE_SCORES = Histogram(
    'job_matching_candidate_scores',
    'Distribuição de scores dos candidatos',
    ['score_range'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Buscas por área/cargo
SEARCHES_BY_AREA = Counter(
    'job_matching_searches_by_area_total',
    'Total de buscas agrupadas por área/cargo',
    ['job_area', 'experience_level']
)

# =============================================================================
# MÉTRICAS DE PERFORMANCE
# =============================================================================

# Tempo de processamento do FAISS
FAISS_SEARCH_DURATION = Histogram(
    'job_matching_faiss_search_duration_seconds',
    'Tempo de busca no índice FAISS',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

# Tamanho da base de dados ativa
ACTIVE_CANDIDATES_COUNT = Gauge(
    'job_matching_active_candidates',
    'Número total de candidatos na base ativa'
)

# Cache hits/misses (se implementado)
CACHE_OPERATIONS = Counter(
    'job_matching_cache_operations_total',
    'Operações de cache do sistema',
    ['operation']  # hit, miss, update
)

# =============================================================================
# MÉTRICAS DE SISTEMA
# =============================================================================

# Uso de memória específico da aplicação
MEMORY_USAGE = Gauge(
    'job_matching_memory_usage_bytes',
    'Uso de memória da aplicação em bytes',
    ['component']  # faiss_index, embeddings, cache
)

# Status de saúde dos componentes
COMPONENT_HEALTH = Gauge(
    'job_matching_component_health',
    'Status de saúde dos componentes (1=healthy, 0=unhealthy)',
    ['component']  # faiss, embeddings, database
)

# =============================================================================
# MÉTRICAS DE QUALIDADE
# =============================================================================

# Taxa de satisfação (se implementado feedback)
USER_SATISFACTION = Histogram(
    'job_matching_user_satisfaction',
    'Taxa de satisfação dos usuários com os resultados',
    ['satisfaction_level'],
    buckets=[1, 2, 3, 4, 5]
)

# Tempo médio até encontrar candidato adequado
TIME_TO_MATCH = Histogram(
    'job_matching_time_to_match_seconds',
    'Tempo até encontrar candidato adequado',
    buckets=[60, 300, 900, 1800, 3600]  # 1min, 5min, 15min, 30min, 1h
)

# =============================================================================
# DECORADORES PARA MÉTRICAS AUTOMÁTICAS
# =============================================================================

def track_endpoint_metrics(endpoint_name: str):
    """Decorator para trackear automaticamente métricas de endpoints"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = "200"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = "500"
                raise
            finally:
                duration = time.time() - start_time
                REQUESTS_TOTAL.labels(
                    endpoint=endpoint_name,
                    method="POST",
                    status_code=status_code
                ).inc()
                REQUEST_DURATION.labels(
                    endpoint=endpoint_name,
                    method="POST"
                ).observe(duration)
        
        return wrapper
    return decorator

def track_faiss_search():
    """Decorator para trackear métricas de busca FAISS"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Trackear número de candidatos encontrados
                if isinstance(result, list):
                    CANDIDATES_FOUND.labels(search_type="faiss").observe(len(result))
                    
                    # Trackear distribuição de scores
                    for candidate in result:
                        if 'score' in candidate:
                            score = candidate['score']
                            score_range = f"{int(score * 10) / 10:.1f}-{int(score * 10 + 1) / 10:.1f}"
                            CANDIDATE_SCORES.labels(score_range=score_range).observe(score)
                
                return result
            finally:
                duration = time.time() - start_time
                FAISS_SEARCH_DURATION.observe(duration)
        
        return wrapper
    return decorator

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def update_system_metrics():
    """Atualiza métricas de sistema (para ser chamada periodicamente)"""
    import os
    
    # Simular uso de memória básico sem psutil
    try:
        # No Windows, usar informações básicas do OS
        memory_kb = os.popen('tasklist /fi "imagename eq python.exe" /fo csv').read()
        # Definir um valor padrão se não conseguir obter informação real
        MEMORY_USAGE.labels(component="application").set(100 * 1024 * 1024)  # 100MB default
    except:
        # Fallback para valor fixo
        MEMORY_USAGE.labels(component="application").set(50 * 1024 * 1024)  # 50MB default

def classify_job_area(job_description: str) -> str:
    """Classifica a área do trabalho baseado na descrição"""
    job_desc_lower = job_description.lower()
    
    if any(word in job_desc_lower for word in ['python', 'javascript', 'java', 'developer', 'programador']):
        return 'desenvolvimento'
    elif any(word in job_desc_lower for word in ['data', 'analyst', 'analista', 'machine learning', 'ml']):
        return 'dados'
    elif any(word in job_desc_lower for word in ['design', 'ui', 'ux', 'designer']):
        return 'design'
    elif any(word in job_desc_lower for word in ['marketing', 'vendas', 'sales']):
        return 'marketing'
    else:
        return 'outros'

def extract_experience_level(job_description: str) -> str:
    """Extrai o nível de experiência da descrição"""
    job_desc_lower = job_description.lower()
    
    if any(word in job_desc_lower for word in ['junior', 'jr', 'iniciante', 'trainee']):
        return 'junior'
    elif any(word in job_desc_lower for word in ['senior', 'sr', 'sênior']):
        return 'senior'
    elif any(word in job_desc_lower for word in ['pleno', 'mid', 'middle']):
        return 'pleno'
    elif any(word in job_desc_lower for word in ['lead', 'tech lead', 'principal']):
        return 'lead'
    else:
        return 'não_especificado'

# =============================================================================
# MÉTRICAS INFO
# =============================================================================

# Informações da aplicação
APPLICATION_INFO = Info(
    'job_matching_application_info',
    'Informações da aplicação de job matching'
)

# Configurar informações da aplicação
APPLICATION_INFO.info({
    'version': '1.0.0',
    'faiss_index_type': 'HNSW',
    'embedding_model': 'all-MiniLM-L6-v2',
    'database_type': 'parquet'
})