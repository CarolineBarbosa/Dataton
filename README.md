# Job Matching System — Datathon FIAP (Decision)

Resumo
Este repositório contém a solução desenvolvida para o Datathon da FIAP com base no estudo de caso da Decision. O objetivo é automatizar e melhorar o processo de recrutamento usando NLP, embeddings e um índice vetorial FAISS para sugerir candidatos relevantes a partir das bases vagas.json, prospects.json e applicants.json.

Contexto do problema
A Decision atua em bodyshop e recrutamento de TI. Hoje o processo exige muito esforço manual dos hunters e sofre com:
- Falta de padronização nas entrevistas;
- Dificuldade em avaliar engajamento e fit cultural rapidamente;
- Tempo elevado para encontrar candidatos adequados.

O que foi construído
1. Pré-processamento dos dados
   - Tratamento e normalização dos arquivos originais (applicants, prospects, vagas).
   - Extração de campos relevantes (experiência, formação, idiomas, skills) e montagem de textos padronizados (CV sintetizado).

2. Geração de embeddings e índice FAISS
   - Criação de embeddings para cada candidato e para descrições de vaga.
   - Indexação com FAISS usando distância por similaridade cosseno.
   - Salvamento do index FAISS e do mapeamento candidato→vetor em arquivos (estes arquivos funcionam como nosso "modelo" em produção).

3. Módulo de recrutador (matching)
   - Módulo que recebe uma vaga, gera seu embedding e realiza retrieval no índice FAISS.
   - Retorna candidatos rankeados pelo score de similaridade.

4. API (FastAPI)
   - Endpoint /predict que recebe texto de descrição da vaga e retorna candidatos recomendados (top-K).
   - Usa os arquivos gerados pelo FAISS para responder em produção.

5. Web app (Streamlit)
   - Interface para interação human-in-the-loop: enviar descrições, ajustar parâmetros, reexecutar buscas e exportar CVs padronizados (experiência, educação, inglês, curso superior).
   - Permite iterações: basta adicionar/ajustar o texto no chat e solicitar nova seleção.

## Boas práticas e notas
- Modelos/índices FAISS são considerados artefatos de produção — versionar e salvar hashes.
- Adicione a indexação sempre que novos candidatos forem adicionados.
- Testes unitários (pytest) e cobertura ≥ 80% recomendados.

## 🛠️ Stack Tecnológica

- **Linguagem**: Python 3.10
- **Bibliotecas de ML/Processamento**: pandas, numpy, scikit-learn, sentence-transformers
- **Banco Vetorial**: FAISS
- **API**: FastAPI
- **Visualização**: Streamlit
- **Serialização**: pickle + faiss index
- **Empacotamento**: Docker
- **Testes**: pytest
- **Deploy**: Local (Docker) — possível extensão para cloud
- **Monitoramento**: Grafana + Prometheus

## 🚀 Início Rápido

### Pré-requisitos
- **Docker** e **Docker Compose** (obrigatório)
- **Git** (para clonar o repositório)

### Setup Único
```bash
# 1. Clone o repositório
git clone <repository-url>
cd Dataton

# 2. Inicie todo o sistema com um comando
docker-compose up --build -d
```

### 🌐 Acessos Web
Aguarde alguns minutos para o build e inicialização, depois acesse:

- **🎯 Interface Streamlit**: http://localhost:8501 *(Principal - Interface Visual)*
- **🚀 API FastAPI**: http://localhost:8000/docs *(Documentação da API)*
- **📊 Prometheus**: http://localhost:9090 *(Métricas)*
- **📈 Grafana**: http://localhost:3000 *(Dashboards - admin/admin123)*

## 🛠️ Comandos Docker

```bash
# ✅ Iniciar sistema completo
docker-compose up --build -d

# 📊 Ver status dos containers
docker-compose ps

# 📋 Ver logs gerais
docker-compose logs

# 🔍 Ver logs de um serviço específico
docker-compose logs streamlit_app
docker-compose logs ml_app

# ⏹️ Parar sistema completo
docker-compose down

# 🔄 Rebuild completo (se houver problemas)
docker-compose down
docker-compose up --build -d
```

## 🗂️ Estrutura do Projeto

```
├── app/                    # API FastAPI
│   ├── main.py            # Endpoints da API
│   └── routes.py          # Rotas organizadas
├── src/                    # Código principal do ML
│   ├── embedding_manager.py
│   ├── recruiter.py       # Lógica de matching
│   ├── preprocessing.py
│   └── ...
├── streamlit_app.py       # 🎯 Interface web principal
├── requirements.txt       # Dependências Python
├── docker-compose.yml     # 🐳 Orquestração dos serviços
├── Dockerfile             # Container da API
├── Dockerfile.streamlit   # Container do Streamlit
└── prometheus/            # Configuração de monitoramento
    └── prometheus.yml
```

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STREAMLIT     │    │   API FASTAPI   │    │   PROMETHEUS    │
│                 │───▶│                 │───▶│                 │
│ Interface Visual│    │ Processa vagas  │    │ Coleta métricas │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   FAISS INDEX   │    │    GRAFANA      │
                       │                 │    │                 │
                       │ Busca candidatos│    │ Visualiza dados │
                       └─────────────────┘    └─────────────────┘
```

### 🎯 Como Funciona o Matching

1. **Entrada**: Descrição da vaga (texto)
2. **Processamento**: 
   - Classifica área de trabalho (desenvolvimento, dados, design...)
   - Extrai nível de experiência (junior, pleno, senior...)
   - Gera embedding (vetor de 384 dimensões)
3. **Busca**: FAISS encontra candidatos similares
4. **Resultado**: Lista de candidatos com scores de compatibilidade

## 🧪 Testando o Sistema

### 1️⃣ Verificar Containers
```bash
docker-compose ps
```
*Deve mostrar 4 serviços rodando: ml_app, streamlit_app, prometheus, grafana*

### 2️⃣ Testar Interface Principal
- Acesse: http://localhost:8501
- Use a interface para fazer matching de candidatos

### 3️⃣ Testar API
- Documentação: http://localhost:8000/docs
- Teste o endpoint `/predict`:
```json
{
  "job_description": "Desenvolvedor Python sênior com experiência em machine learning",
  "top_n": 5,
  "search_k": 50
}
```
- Endpoint `/metrics` para métricas Prometheus
- Endpoint `/health` para status da aplicação

### 4️⃣ Verificar Monitoramento
**Prometheus** (http://localhost:9090):
- Vá em **Status → Targets**
- Verifique se `ml_app:8000` está **UP**
- Teste query: `job_matching_requests_total`

**Grafana** (http://localhost:3000):
- Login: **admin / admin123**
- Data Source já configurado: `http://prometheus:9090`
- Dashboard pré-configurado disponível

## 📊 Métricas e Monitoramento

### Métricas Coletadas Automaticamente:
- **Performance**: Taxa de requisições, latência, tempo de busca FAISS
- **Negócio**: Candidatos encontrados, scores de matching, áreas mais buscadas
- **Sistema**: Saúde dos componentes, uso de memória, status da aplicação

### Queries Úteis para Grafana:
```prometheus
# Total de requisições
job_matching_requests_total

# Taxa de requisições por minuto
rate(job_matching_requests_total[1m])

# Candidatos encontrados
job_matching_candidates_found_sum

# Buscas por área
job_matching_searches_by_area_total

# Duração média das requisições
job_matching_request_duration_seconds_sum / job_matching_request_duration_seconds_count
```

## 🔧 Troubleshooting

### Problema: Containers não iniciam
```bash
# Ver logs detalhados
docker-compose logs

# Limpar e tentar novamente
docker-compose down
docker-compose up --build -d
```

### Problema: ImportError com huggingface_hub
Se aparecer erro `cannot import name 'cached_download' from 'huggingface_hub'`:
- ✅ **Já corrigido**: O projeto usa versões compatíveis fixas
- `sentence-transformers==3.1.1` 
- `huggingface-hub==0.19.4`

### Problema: Porta ocupada
```bash
# Verificar portas em uso no Windows
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Parar outros serviços ou alterar portas no docker-compose.yml
```

### Problema: Build muito lento
- O primeiro build pode levar varios minutos (download de modelos ML)
- Builds subsequentes são mais rápidos (cache do Docker)
- Use `docker-compose up --build -d` para rebuild otimizado

## 📋 Logs e Monitoramento

```bash
# Ver status geral
docker-compose ps

# Logs em tempo real
docker-compose logs -f

# Logs de um serviço específico
docker-compose logs -f streamlit_app
docker-compose logs -f ml_app
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License.