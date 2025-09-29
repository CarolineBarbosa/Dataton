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

from src.faiss_artifact_creator import orchestrate_faiss_creation


if __name__ == "__main__":
    orchestrate_faiss_creation()


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

#     # TODO: criar um notebook com exemplos de uso do job-matchin e avaliar com base nos prospects reais
# from src.evaluate import *
# if __name__ == "__main__":
#     main()
 