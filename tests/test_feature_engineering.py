import pytest
import pandas as pd
from src.feature_engineering import combine_columns, filter_columns, extract_filters_from_text

def test_combine_columns():
    data = {
        'col1': ['value1', 'value2', None],
        'col2': ['value3', None, 'value4'],
        'col3': [None, 'value5', 'value6']
    }
    df = pd.DataFrame(data)
    columns_to_combine = ['col1', 'col2', 'col3']
    result = combine_columns(df, columns_to_combine)
    expected_text = [
        'col1: value1 col2: value3',
        'col1: value2 col3: value5',
        'col2: value4 col3: value6'
    ]
    assert result['text'].tolist() == expected_text

def test_filter_columns():
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    columns_to_keep = ['col1', 'col3']
    result = filter_columns(df, columns_to_keep)
    assert list(result.columns) == columns_to_keep
    assert result.equals(df[columns_to_keep])

def test_extract_filters_from_text():
    text = "This is a job for a Senior professional with Graduação level education in São Paulo. Requires ingles."
    filters = extract_filters_from_text(text)
    expected_filters = {
        'Senior': 1,
        'Graduação': 1,
        'São Paulo': 1,
        'nivel_ingles': 1
    }
    assert filters == expected_filters

def test_extract_filters_from_text_no_match():
    text = "This text does not match any filters."
    filters = extract_filters_from_text(text)
    assert filters == {}

def test_extract_filters_from_text_partial_match():
    text = "Looking for a Junior professional in Rio de Janeiro."
    filters = extract_filters_from_text(text)
    expected_filters = {
        'Junior': 1,
        'Rio de Janeiro': 1
    }
    assert filters == expected_filters