import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pydantic import BaseModel
from utils import load_parquet
import yaml

class EmbeddingModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.model_name = config['embedding_model']['name']
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)


def combine_columns(df, columns_to_combine):
    """Combine specified columns into a single text column."""
    df['text'] = df[columns_to_combine].fillna('').apply(
        lambda row: ' '.join([
            f"{col}: {row[col]}" for col in columns_to_combine
        ]),
        axis=1
    )
    return df

def filter_columns(df, columns_to_keep):
    """Filter the dataframe to keep only the specified columns."""
    return df[columns_to_keep]

def generate_embeddings(df, text_column, model_config_path='models_config.yaml'):
    """Generate embeddings for the text column using a pre-trained model."""
    embedding_model = EmbeddingModel(config_path=model_config_path)
    df['embeddings'] = list(embedding_model.encode(
        df[text_column].tolist(), 
        normalize_embeddings=True
    ))
    embeddings = embedding_model.encode(
        df[text_column].tolist(), 
        normalize_embeddings=True
    )
    df['embeddings'] = embeddings
    return df

def process_entity(file_path, columns_to_combine, columns_to_keep):
    """Process an entity by loading data, combining columns, and generating embeddings."""
    df = load_parquet(file_path)
    df = combine_columns(df, columns_to_combine)
    df = filter_columns(df, columns_to_keep)
    df = generate_embeddings(df, 'text')
    return df
