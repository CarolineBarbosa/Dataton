import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np

# FAISS: install with `pip install faiss-cpu` on macOS/linux (or conda install -c pytorch faiss-cpu)
import faiss

class FAISSIndexer:
    """
    Simple FAISS indexer using IndexFlatIP (inner product) for cosine search.
    Embeddings should already be normalized (so IP ~= cosine similarity).
    Stores metadata mapping (int id -> metadata) in a pickle.
    """
    def __init__(self, config: Dict[str, Any]):
        self.index_path = config.get("paths", {}).get("index_path", "src/data/faiss.index")
        self.meta_path = config.get("paths", {}).get("meta_path", "src/data/faiss_meta.pkl")
        self.index_type = config.get("index", {}).get("index_type", "flat")
        self.k_default = config.get("index", {}).get("k", 5)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, Any] = {}
        self.next_id = 0

        self._load()

    def _init_index(self, dim: int):
        # Use IndexFlatIP so you can use normalized embeddings for cosine-like similarity
        if self.index_type == "flat":
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            # fallback to flat
            self.index = faiss.IndexFlatIP(dim)

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception:
                self.index = None
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", {})
                    self.next_id = max(self.metadata.keys()) + 1 if self.metadata else 0
            except Exception:
                self.metadata = {}
                self.next_id = 0

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"metadata": self.metadata}, f)

    def add_embedding(self, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        embedding: 1D numpy array (float32) or 2D (n, dim). If 1D, adds single vector.
        metadata: optional dict to associate with this vector.
        Returns assigned id(s).
        """
        if embedding.ndim == 1:
            emb = embedding.reshape(1, -1)
        else:
            emb = embedding

        emb = emb.astype(np.float32)
        n, dim = emb.shape

        if self.index is None:
            self._init_index(dim)

        # Assign ids and add metadata
        ids = []
        for i in range(n):
            assigned_id = self.next_id
            self.metadata[assigned_id] = metadata if metadata is not None else {}
            ids.append(assigned_id)
            self.next_id += 1

        # FAISS IndexFlat does not support supplying explicit ids; appends in order.
        self.index.add(emb)
        self._save()
        return ids if len(ids) > 1 else ids[0]

    def query_embedding(self, embedding: np.ndarray, k: Optional[int] = None, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Returns list of dicts: [{id, score, metadata}, ...]
        Filters can be applied to metadata: e.g., {"column_name": "value"}.
        """
        if k is None:
            k = self.k_default
        if self.index is None or self.index.ntotal == 0:
            return []

        emb = embedding.astype(np.float32).reshape(1, -1)
        # Filter embeddings to only include those with metadata "source" equal to "applicants"
        valid_ids = [idx for idx, meta in self.metadata.items() if meta.get("source") == "applicants"]
        if not valid_ids:
            return []

        # Create a mask for valid indices
        valid_indices = np.array(valid_ids, dtype=np.int64)
        self.index.remove_ids(np.setdiff1d(np.array(list(self.metadata.keys()), dtype=np.int64), valid_indices))

        D, I = self.index.search(emb, 100)  # D: scores, I: indices (position-based)

        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
         
            if idx < 0:
                continue
            metadata = self.metadata.get(idx, {})
            # Simplify metadata columns
            metadata = transform_metadata(metadata)

            # # Apply filters if provided
            # if filters:
            #     match = all(metadata.get(key) == value for key, value in filters.items())
            #     if not match:
            #         continue
            #     results.append({"id": int(idx), "score": float(score), "metadata": metadata})

        if not results and len(results) == 0:  # If no results after filtering
            for score, idx in zip(D[0].tolist(), I[0].tolist()):

                D, I = self.index.search(emb, k)  # Re-run search without filters
                for score, idx in zip(D[0].tolist(), I[0].tolist()):
                    if idx < 0:
                        continue
                    metadata = self.metadata.get(idx, {})
                    results.append({"id": int(idx), "score": float(score), "metadata": metadata})
                
        return results[:k]

def transform_metadata(metadata):
    def _has_value(v):
        if v is None:
            return False
        # sequences/collections/strings: consider non-empty -> True
        if isinstance(v, (str, list, tuple, set, dict)):
            return len(v) > 0
        # numbers/booleans/other truthy values
        try:
            return bool(v)
        except Exception:
            return False

    if "nivel_ingles" in metadata:
        metadata["nivel_ingles"] = 1 if _has_value(metadata["nivel_ingles"]) else 0
    if "nivel_espanhol" in metadata:
        metadata["nivel_espanhol"] = 1 if _has_value(metadata["nivel_espanhol"]) else 0
        # Create dummy columns for 'nivel profissional'
    if "nivel_profissional" in metadata:
        dummy_profissional = {"Junior": 0, "Pleno": 0, "Senior": 0}
        if metadata["nivel_profissional"] in dummy_profissional:
            dummy_profissional[metadata["nivel_profissional"]] = 1
        metadata.update(dummy_profissional)

        # Create dummy columns for 'nivel academico'
    if "nivel_academico" in metadata:
        dummy_academico = {
                "Ensino Médio": 0,
                "Graduação": 0,
                "Pós-Graduação": 0,
                "Mestrado": 0,
                "Doutorado": 0,
            }
        if metadata["nivel_academico"] in dummy_academico:
            dummy_academico[metadata["nivel_academico"]] = 1
        metadata.update(dummy_academico)

        # Create dummy columns for 'cidade'
    if "cidade" in metadata:
        dummy_cidade = {
                "São Paulo": 0,
                "Rio de Janeiro": 0,
                "Belo Horizonte": 0,
                "Curitiba": 0,
                "Porto Alegre": 0,
                "Salvador": 0,
                "Brasília": 0,
                "Fortaleza": 0,
                "Recife": 0,
                "Manaus": 0,
            }
        if metadata["cidade"] in dummy_cidade:
            dummy_cidade[metadata["cidade"]] = 1
        metadata.update(dummy_cidade)
    return metadata

def add_entity_embeddings_to_faiss(df, emb_mgr, indexer):
    for i, row in df.iterrows():
        text = row['text']
        vec = emb_mgr.generate_embedding(text)
        metadata = row.to_dict()  # Convert all columns of the row to a dictionary
        metadata.update({"source": "applicants", "idx": i})  # Add additional metadata
        assigned_id = indexer.add_embedding(vec, metadata=metadata)


