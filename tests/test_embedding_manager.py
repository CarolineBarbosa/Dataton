import pytest
import numpy as np
from src.embedding_manager import EmbeddingManager

@pytest.fixture
def embedding_manager():
    return EmbeddingManager(config_path="models_config.yaml")

def test_generate_embedding_single_text(embedding_manager):
    text = "This is a test sentence."
    embedding = embedding_manager.generate_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # 1D array for single text
    assert embedding.size > 0  # Ensure the embedding is not empty

def test_generate_embedding_list_of_texts(embedding_manager):
    texts = ["This is the first sentence.", "This is the second sentence."]
    embeddings = embedding_manager.generate_embedding(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2  # 2D array for list of texts
    assert embeddings.shape[0] == len(texts)  # Number of embeddings matches input texts
    assert embeddings.shape[1] > 0  # Ensure embeddings have dimensions

def test_generate_embedding_invalid_input(embedding_manager):
    invalid_input = 12345  # Not a string or list of strings
    with pytest.raises(TypeError):
        embedding_manager.generate_embedding(invalid_input)