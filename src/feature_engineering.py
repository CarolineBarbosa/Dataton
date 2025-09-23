from utils import load_parquet
import numpy as np
import pandas as pd


def combine_columns(df, columns_to_combine):
    """Combine specified columns into a single text column."""
    df['text'] = df[columns_to_combine].apply(
        lambda row: ' '.join([
            f"{col}: {row[col]}" for col in columns_to_combine if pd.notnull(row[col])
        ]),
        axis=1
    )
    return df

def filter_columns(df, columns_to_keep):
    """Filter the dataframe to keep only the specified columns."""
    return df[columns_to_keep]


def process_entity(file_path, columns_to_combine, columns_to_keep):
    """Process an entity by loading data, combining columns, and generating embeddings."""
    df = load_parquet(file_path)
    df = combine_columns(df, columns_to_combine)
    df = filter_columns(df, columns_to_keep)
    return df
