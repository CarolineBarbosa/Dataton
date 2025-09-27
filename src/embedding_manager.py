from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml


class EmbeddingManager:
    """
    Thin wrapper around EmbeddingModel in src/feature_engineering.py.
    Returns numpy arrays (1D for single text, 2D for list).
    """
    def __init__(self, config_path: str = "models_config.yaml"):
        self.model = EmbeddingModel(config_path)

    def generate_embedding(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            vec = self.model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        elif isinstance(text, list):
            vecs = self.model.encode(text, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        else:
            raise TypeError("text must be str or list[str]")
        

class EmbeddingModel:
    def __init__(self, config_path):

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.model_name = config['embedding_model']['name']
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)


