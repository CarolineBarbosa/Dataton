from typing import List, Union
import numpy as np
from src.feature_engineering import EmbeddingModel

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