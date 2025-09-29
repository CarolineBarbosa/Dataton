from utils import * 
# from utils.datasource_config import DatasourceConfig
# from src.preprocessing import preprocessing
from src.feature_engineering import process_entity, extract_filters_from_text, prepare_vagas_applicants_data
from typing import List, Dict, Any, Optional
import os
import sys
import yaml


# Make workspace root importable so `from src.*` works when running this script
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, workspace_root)
from src.embedding_manager import EmbeddingManager

from src.indexer import FAISSIndexer, add_entity_embeddings_to_faiss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_applicants_data(file_path: str, columns_to_combine: List[str], columns_to_keep: List[str]) -> Any:
    """Prepare applicants data by processing the entity."""
    print("Preparing applicants data...")
    df_applicants = process_entity(file_path, columns_to_combine, columns_to_keep)
    print("Finished preparing applicants data.")
    return df_applicants


def initialize_components(index_config_path: str, models_config_path: str) -> (EmbeddingManager, FAISSIndexer):
    """Initialize the embedding manager and FAISS indexer."""
    print("Initializing components...")
    index_config = load_config(index_config_path)
    emb_mgr = EmbeddingManager(config_path=models_config_path)
    indexer = FAISSIndexer(index_config)
    print("Components initialized.")
    return emb_mgr, indexer


def add_applicants_to_faiss(df_applicants: Any, emb_mgr: EmbeddingManager, indexer: FAISSIndexer) -> None:
    """Add applicants embeddings to the FAISS index."""
    print("Starting to add applicants embeddings to FAISS index...")
    add_entity_embeddings_to_faiss(df_applicants, emb_mgr, indexer)
    print("Finished adding applicants embeddings to FAISS index.")


def orchestrate_faiss_creation() -> None:
    """Orchestrate the FAISS artifact creation process."""
    # Define file paths and columns
    applicants_file_path = "data/processed/applicants.parquet"
    applicants_columns_to_combine = [
        'titulo_profissional', 'objetivo_profissional', 'area_atuacao',
        'conhecimentos_tecnicos', 'certificacoes', 'nivel_profissional',
        'nivel_academico', 'cursos', 'cv_pt',
        'nivel_ingles', 'nivel_espanhol'
    ]
    applicants_columns_to_keep = [
        'applicants_id', 'titulo_profissional', 'nivel_profissional',
        'nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'local', 'text'
    ]
    index_config_path = os.path.join("src", "config", "index_config.yaml")
    models_config_path = os.path.join("models_config.yaml")

    # Process applicants data
    df_applicants = prepare_applicants_data(applicants_file_path, applicants_columns_to_combine, applicants_columns_to_keep)

    # Initialize components
    emb_mgr, indexer = initialize_components(index_config_path, models_config_path)

    # Add embeddings to FAISS index
    add_applicants_to_faiss(df_applicants, emb_mgr, indexer)

