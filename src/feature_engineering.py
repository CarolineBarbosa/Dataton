from src.utils import load_parquet
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

def prepare_vagas_applicants_data(vagas_file_path, applicants_file_path, vagas_columns_to_combine, applicants_columns_to_combine, vagas_columns_to_keep, applicants_columns_to_keep):
    df_vagas = process_entity(vagas_file_path, vagas_columns_to_combine, vagas_columns_to_keep)

    # Process applicants
    df_applicants = process_entity(applicants_file_path, applicants_columns_to_combine, applicants_columns_to_keep)
    return df_vagas,df_applicants

def extract_filters_from_text(text):
    filters = {}
    keywords = {
        'nivel_profissional': ['Junior', 'Pleno', 'Senior'],
        'nivel_academico': ['Ensino Médio', 'Graduação', 'Pós-Graduação', 'Mestrado', 'Doutorado'],
        'nivel_ingles': ['ingles'],
        'nivel_espanhol': ['espanhol'],
        'cidade': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Curitiba', 
                    'Porto Alegre', 'Salvador', 'Brasília', 'Fortaleza', 'Recife', 'Manaus']
    }

    for key, values in keywords.items():
        if key in ['nivel_ingles', 'nivel_espanhol']:
            if any(value.lower() in text.lower() for value in values):
                filters[key] = 1
        elif key in ['nivel_profissional', 'nivel_academico', 'cidade']:
            for value in values:
                if value.lower() in text.lower():
                    filters[f"{value}"] = 1
                    break

    return filters