import os
import sys
import yaml
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="🎯 Job Matching System",
    page_icon="🎯",
    layout="wide"
)

# Adicionar ao path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

# URL da API - Use o nome do serviço Docker para comunicação entre containers
API_URL = "http://ml_app:8000/predict"

def main():
    st.title("🎯 Sistema de Job Matching")
    st.markdown("### Encontre os melhores candidatos para suas vagas!")
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        top_n = st.slider("Número de candidatos", min_value=1, max_value=20, value=5)
        search_k = st.slider("Amplitude da busca", min_value=10, max_value=200, value=50)
        
        # Status da API
        st.subheader("📡 Status da API")
        try:
            response = requests.get("http://ml_app:8000/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Online")
            else:
                st.error("❌ API com problemas")
        except:
            st.error("❌ API Offline")
            st.warning("Certifique-se de que a API está rodando: `docker-compose up -d`")

    # Input principal
    st.header("📝 Descrição da Vaga")
    job_description = st.text_area(
        "Digite a descrição da vaga ou habilidades desejadas:",
        placeholder="Ex: Desenvolvedor Python com experiência em machine learning, pandas, scikit-learn...",
        height=150
    )
    
    # Exemplos rápidos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🐍 Data Scientist"):
            job_description = "Cientista de dados Python, machine learning, pandas, scikit-learn, análise estatística, TensorFlow"
            st.experimental_rerun()
    
    with col2:
        if st.button("⚛️ Frontend Developer"):
            job_description = "Desenvolvedor Frontend React, TypeScript, JavaScript, CSS, HTML, responsivo, UX/UI"
            st.experimental_rerun()
    
    with col3:
        if st.button("🔧 DevOps Engineer"):
            job_description = "DevOps AWS, Docker, Kubernetes, Terraform, Jenkins, CI/CD, automação, infraestrutura"
            st.experimental_rerun()

    # Botão de busca
    if st.button("🔍 Buscar Candidatos", type="primary"):
        if not job_description.strip():
            st.warning("⚠️ Por favor, insira uma descrição da vaga!")
            return
            
        with st.spinner("🔍 Buscando candidatos..."):
            try:
                # Fazer requisição para a API
                payload = {
                    "job_description": job_description,
                    "top_n": top_n,
                    "search_k": search_k
                }
                
                response = requests.post(API_URL, json=payload, timeout=30)
                
                if response.status_code == 200:
                    candidates = response.json()
                    
                    if candidates:
                        st.success(f"✅ Encontrados {len(candidates)} candidatos!")
                        
                        # Mostrar resultados
                        st.header("🏆 Top Candidatos")
                        
                        for i, candidate in enumerate(candidates, 1):
                            with st.expander(f"#{i} - Score: {candidate.get('score', 0):.4f}"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**ID:** {candidate.get('applicant_id', 'N/A')}")
                                    st.write(f"**Nome:** {candidate.get('nome', 'N/A')}")
                                    
                                    # Metadata
                                    metadata = candidate.get('metadata', {})
                                    if metadata:
                                        st.write("**Informações Adicionais:**")
                                        for key, value in metadata.items():
                                            if value and str(value) != 'nan':
                                                st.write(f"- **{key}:** {value}")
                                
                                with col2:
                                    # Score visual
                                    score_pct = candidate.get('score', 0) * 100
                                    st.metric("Score de Compatibilidade", f"{score_pct:.1f}%")
                    else:
                        st.info("🤷‍♂️ Nenhum candidato encontrado para essa descrição. Tente termos mais gerais.")
                        
                else:
                    st.error(f"❌ Erro na API: {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏰ Timeout - A busca demorou muito. Tente reduzir o 'search_k'.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Erro de conexão. Verifique se a API está rodando!")
                st.info("Execute: `docker-compose up -d`")
            except Exception as e:
                st.error(f"❌ Erro inesperado: {e}")

    # Informações adicionais
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        ### 📖 Instruções:
        
        1. **📝 Digite a descrição** da vaga ou habilidades desejadas
        2. **⚙️ Configure** o número de candidatos e amplitude da busca
        3. **🔍 Clique em "Buscar"** para encontrar os melhores matches
        
        ### 🎯 Dicas para melhores resultados:
        
        - ✅ **Seja específico**: "Python machine learning pandas" em vez de "programador"
        - ✅ **Inclua tecnologias**: "React TypeScript Node.js" 
        - ✅ **Mencione senioridade**: "sênior", "júnior", "pleno"
        - ✅ **Use termos técnicos**: nomes de ferramentas, frameworks, linguagens
        
        ### 🔧 Parâmetros:
        
        - **Top N**: Quantos candidatos retornar (1-20)
        - **Search K**: Amplitude da busca inicial (maior = mais opções, mais lento)
        """)

    # Footer
    st.markdown("---")
    st.markdown("🚀 **Sistema de Job Matching** | 📊 [Grafana Dashboard](http://localhost:3000) | 🔍 [API Docs](http://localhost:8000/docs)")

if __name__ == "__main__":
    main()