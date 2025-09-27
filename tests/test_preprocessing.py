import pandas as pd
import pytest

# tests/test_preprocessing.py

import src.preprocessing as preprocessing


def test_ingest_data_calls_load_and_save(monkeypatch):
    # Prepare a dummy dataframe to be returned by load_json
    dummy_df = pd.DataFrame({"col": [1, 2, 3]})
    file_config = {"path": "data/raw/applicants.json"}

    called = {"load_called": False, "save_called": False, "save_args": None}

    def fake_load_json(cfg):
        # ensure the same config is forwarded
        assert cfg is file_config
        called["load_called"] = True
        return dummy_df

    def fake_save_to_parquet(df_arg, name_arg):
        called["save_called"] = True
        called["save_args"] = (df_arg, name_arg)

    monkeypatch.setattr(preprocessing, "load_json", fake_load_json)
    monkeypatch.setattr(preprocessing, "save_to_parquet", fake_save_to_parquet)

    # Call the function under test
    result = preprocessing.ingest_data(file_config)

    # ingest_data returns None explicitly in the implementation
    assert result is None
    assert called["load_called"] is True
    assert called["save_called"] is True

    saved_df, saved_name = called["save_args"]
    # verify the DataFrame passed through unchanged and name is derived from filename
    pd.testing.assert_frame_equal(saved_df, dummy_df)
    assert saved_name == "applicants"


def test_preprocessing_iterates_entities_and_calls_ingest(monkeypatch):
    # Prepare a datasource config with two entities
    datasource_config = {
        "applicants": {"path": "data/raw/applicants.json"},
        "jobs": {"path": "data/raw/jobs.json"},
    }

    calls = []

    def fake_ingest_data(entity_config):
        # record that ingest_data was invoked with the correct config
        calls.append(entity_config)

    monkeypatch.setattr(preprocessing, "ingest_data", fake_ingest_data)

    # Call preprocessing which should call ingest_data for each entity
    preprocessing.preprocessing(datasource_config)

    # Ensure ingest_data was called twice with the expected configs (order preserved by dict in modern Python)
    assert len(calls) == 2
    assert calls[0] == datasource_config["applicants"]
    assert calls[1] == datasource_config["jobs"]