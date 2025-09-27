import pytest
from unittest.mock import MagicMock
import sys
from typing import Any
import pandas as pd
import os
sys.path.append('../')
from src.recruiter import RecruiterBot, find_top_applicants_with_filters
import sys

@pytest.fixture
def mock_embedding_manager():
    emb_mgr = MagicMock()
    emb_mgr.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return emb_mgr

@pytest.fixture
def mock_faiss_indexer():
    indexer = MagicMock()
    indexer.query_embedding.return_value = [
        {"metadata": {"idx": 0}, "score": 0.95},
        {"metadata": {"idx": 1}, "score": 0.90},
    ]
    return indexer

@pytest.fixture
def mock_applicants_df(mocker):
    mock_df = mocker.patch("pandas.read_parquet")
    mock_df.return_value = pd.DataFrame({
        "applicants_id": [101, 102],
        "nome": ["Alice", "Bob"]
    })
    return mock_df

def test_find_top_applicants_with_filters(mock_embedding_manager: MagicMock, mock_faiss_indexer: MagicMock, mock_applicants_df: Any):
    job_description = "Looking for a data scientist with Python skills"
    results = find_top_applicants_with_filters(
        job_description=job_description,
        faiss_indexer=mock_faiss_indexer,
        emb_mgr=mock_embedding_manager,
        filters=None,
        top_n=2
    )
    assert len(results) == 2
    assert results[0]["applicant_id"] == 101
    assert results[0]["nome"] == "Alice"
    assert results[0]["score"] == 0.95

def test_recruiter_bot_chat(mock_embedding_manager: MagicMock, mock_faiss_indexer: MagicMock, mock_applicants_df: Any):
    bot = RecruiterBot(mock_embedding_manager, mock_faiss_indexer)
    message = "Looking for a senior engineer in São Paulo with advanced English"
    reply = bot.chat(message, top_n=2)
    assert "Top matching applicants:" in reply
    assert "1. Alice — Score: 0.9500" in reply
    assert "2. Bob — Score: 0.9000" in reply

def test_recruiter_bot_no_matches(mock_embedding_manager: MagicMock, mock_faiss_indexer: MagicMock, mocker):
    mock_faiss_indexer.query_embedding.return_value = []
    mocker.patch("pandas.read_parquet", return_value=pd.DataFrame())
    bot = RecruiterBot(mock_embedding_manager, mock_faiss_indexer)
    message = "Looking for a role that doesn't exist"
    reply = bot.chat(message, top_n=2)
    assert reply == "No applicants matched the current criteria."

def test_recruiter_bot_empty_message(mock_embedding_manager: MagicMock, mock_faiss_indexer: MagicMock):
    bot = RecruiterBot(mock_embedding_manager, mock_faiss_indexer)
    reply = bot.chat("", top_n=2)
    assert reply == "Please provide a job description or more details."