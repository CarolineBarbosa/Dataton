from utils import * 
# from utils.datasource_config import DatasourceConfig
# from src.preprocessing import preprocessing
from src.feature_engineering import process_entity, extract_filters_from_text, prepare_vagas_applicants_data
from src.indexer import transform_metadata, add_entity_embeddings_to_faiss
from typing import List, Dict, Any, Optional
import os
import sys
import yaml


# Make workspace root importable so `from src.*` works when running this script
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, workspace_root)
from src.embedding_manager import EmbeddingManager
from src.indexer import FAISSIndexer

from src.recruiter import RecruiterBot




# if __name__ == "__main__":
    # Call the pipeline function
    # config = DatasourceConfig("data_source_config.yaml").config
    # preprocessing(config)


    # Define the columns and file paths for vagas and applicants
    # vagas_file_path = "data/processed/vagas.parquet"
    # applicants_file_path = "data/processed/applicants.parquet"

    # vagas_columns_to_combine = [
    #     'titulo_vaga', 'nivel profissional', 'nivel_academico',
    #     'nivel_ingles', 'nivel_espanhol', 'areas_atuacao',
    #     'principais_atividades', 'competencia_tecnicas_e_comportamentais',
    #     'habilidades_comportamentais_necessarias'
    # ]

    # applicants_columns_to_combine = [
    #     'titulo_profissional', 'objetivo_profissional', 'area_atuacao',
    #     'conhecimentos_tecnicos', 'certificacoes', 'nivel_profissional',
    #     'nivel_academico', 'cursos', 'cv_pt',
    #     'nivel_ingles', 'nivel_espanhol'
    # ]

    # vagas_columns_to_keep = ['jobs_id', 'titulo_vaga', 'nivel profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol','cidade','text']
    # applicants_columns_to_keep = ['applicants_id', 'titulo_profissional', 'nivel_profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol','local','text']
    
    # # # Process vagas
    # print("Preparing vagas and applicants data...")
    # df_vagas, df_applicants = prepare_vagas_applicants_data(vagas_file_path, applicants_file_path, vagas_columns_to_combine, applicants_columns_to_combine, vagas_columns_to_keep, applicants_columns_to_keep)
    # print("Finished preparing vagas and applicants data.")
    # index_config_path = os.path.join( "src", "config", "index_config.yaml")
    # models_config_path = os.path.join( "models_config.yaml")

    # # Load index config
    # with open(index_config_path, "r") as f:
    #     index_config = yaml.safe_load(f)

    # # Init components
    # emb_mgr = EmbeddingManager(config_path=models_config_path)
    # indexer = FAISSIndexer(index_config)

    # print("Starting to add embeddings to FAISS index...")
    # # Add embeddings for vagas to FAISS index
    # add_entity_embeddings_to_faiss(df_vagas, emb_mgr, indexer)
    # print("Finished adding vagas embeddings to FAISS index.")
    # # Add embeddings for applicants to FAISS index
    # print("Starting to add applicants embeddings to FAISS index...")
    # add_entity_embeddings_to_faiss(df_applicants, emb_mgr, indexer)
    # print("Finished adding applicants embeddings to FAISS index.")




# Minimal example (uncomment to run as script)
# if __name__ == "__main__":
#     import os, sys, yaml
#     workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#     sys.path.insert(0, workspace_root)
#     with open(os.path.join( "src", "config", "index_config.yaml")) as f:
#         index_cfg = yaml.safe_load(f)
#     emb_mgr = EmbeddingManager(config_path=os.path.join( "models_config.yaml"))
#     indexer = FAISSIndexer(index_cfg)
#     # load your applicants_df here (parquet/csv)
#     bot = RecruiterBot( emb_mgr, indexer)
#     # print(bot.chat("Senior Engenheiro de software"))
#     # print(bot.chat("Cientista de dados com inglês avançado em São Paulo"))
#     print(bot.chat("SAP ABAP Pleno"))

    # TODO: criar um notebook com exemplos de uso do job-matchin e avaliar com base nos prospects reais
from src.evaluate import *
if __name__ == "__main__":
    main()
 