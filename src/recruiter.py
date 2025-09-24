from typing import Dict, List, Any, Optional
import re
import numpy as np
import pandas as pd
import os
import sys
# workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.insert(0, workspace_root)
from src.feature_engineering import extract_filters_from_text
from src.embedding_manager import EmbeddingManager
from src.indexer import FAISSIndexer

def retrieve_top_applicants(emb_mgr: EmbeddingManager,
                            indexer:FAISSIndexer, 
                            query_text:str, 
                            k_top_applicants:int = 5)->List[Dict[str,Any]]:
    filters = extract_filters_from_text(query_text)
    qvec = emb_mgr.generate_embedding(query_text)
    results = indexer.query_embedding(qvec, filters=filters, k=k_top_applicants)
    return results

def find_top_applicants_with_filters(
    job_description: str,
    faiss_indexer: FAISSIndexer,
    emb_mgr: EmbeddingManager,
    filters: Optional[Dict[str, bool]] = None,
    top_n: int = 5,
    search_k: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve candidate ids from FAISS by querying with the job_description embedding,
    then apply column-level include/exclude filters against applicants_df and return
    the top_n matches.

    - search_k: number of neighbors to fetch from FAISS (fetch more and then filter down).
    - filters: dict with keys like "col:val" and boolean flag True=include, False=exclude.
    """
    applicants_df = pd.read_parquet("data/processed/applicants.parquet")

    if applicants_df is None or len(applicants_df) == 0:
        return []
    raw_results = retrieve_top_applicants(
        emb_mgr=emb_mgr,
        indexer=faiss_indexer,
        query_text=job_description,
        k_top_applicants=top_n)

    # apply filtering and collect candidates
    candidates = []
    for r in raw_results:
        meta = r.get("metadata", {})

        # meta should include 'idx' pointing to row in applicants_df
        idx = meta.get("idx")
        if idx is None:
            continue
        try:
            row = applicants_df.loc[idx]
        except Exception:
            # fallback to iloc if loc fails (e.g., index is simple range)
            try:
                row = applicants_df.iloc[idx]
            except Exception:
                continue

        candidates.append({
            "applicant_idx": int(idx),
            "applicant_id": row.get("applicants_id", None),
            "nome": row.get("nome", None) if "nome" in row.index else None,
            "score": float(r.get("score", 0.0)),
            "metadata": meta
        })

    # sort by score desc and take top_n
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
    return candidates


class RecruiterBot:
    """
    Lightweight recruiter bot that keeps a running job_description and filters,
    encodes queries, retrieves candidates from FAISS and returns a textual response.

    Usage:
      emb_mgr = EmbeddingManager(config_path="models_config.yaml")
      indexer = FAISSIndexer(index_config)
      bot = RecruiterBot(applicants_df, emb_mgr, indexer)
      reply = bot.chat("Looking for a senior engineer in São Paulo with advanced English")
    """
    def __init__(self, emb_mgr: EmbeddingManager, faiss_indexer: FAISSIndexer):
        self.job_description = ""
        self.filters: Dict[str, bool] = {}
        self.emb_mgr = emb_mgr
        self.faiss_indexer = faiss_indexer

    def chat(self, message: str, top_n: int = 5) -> str:
        """Update state and return a short textual reply with top matches."""
        self.job_description = message
        if not self.job_description:
            return "Please provide a job description or more details."

        matches = find_top_applicants_with_filters(
            job_description=self.job_description,
            faiss_indexer=self.faiss_indexer,
            emb_mgr=self.emb_mgr,
            filters=self.filters,
            top_n=top_n,
            search_k=200
        )

        if not matches:
            return "No applicants matched the current criteria."

        reply_lines = ["Top matching applicants:"]
        for i, m in enumerate(matches, start=1):
            name = m.get("nome") or f"idx:{m['applicant_idx']}"
            reply_lines.append(f"{i}. {name} — Score: {m['score']:.4f} (row_idx={m['applicant_idx']})")
        return "\n".join(reply_lines)


# Minimal example (uncomment to run as script)
if __name__ == "__main__":
    import os, sys, yaml
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, workspace_root)
    with open(os.path.join(workspace_root, "src", "config", "index_config.yaml")) as f:
        index_cfg = yaml.safe_load(f)
    emb_mgr = EmbeddingManager(config_path=os.path.join(workspace_root, "models_config.yaml"))
    indexer = FAISSIndexer(index_cfg)
    # load your applicants_df here (parquet/csv)
    bot = RecruiterBot( emb_mgr, indexer)
    print(bot.chat("Senior software engineer in São Paulo with advanced English"))