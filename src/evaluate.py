import os
import sys
import yaml
import pandas as pd
from src.feature_engineering import combine_columns
from src.recruiter import find_top_applicants_with_filters
from src.embedding_manager import calculate_similarity, EmbeddingManager
from src.indexer import FAISSIndexer
from src.recruiter import RecruiterBot


def load_data():
    """Load and preprocess data."""
    df_prospects = pd.read_parquet('data/processed/prospects.parquet')
    df_applicants = pd.read_parquet('data/processed/applicants.parquet')
    df_jobs = pd.read_parquet('data/processed/vagas.parquet')
    return df_prospects, df_applicants, df_jobs


def preprocess_prospects(df):
    """Preprocess prospects data."""
    df['codigo_list'] = df['prospects'].apply(lambda x: [item['codigo'] for item in x])
    df['situacao_candidato'] = df['prospects'].apply(lambda x: [item['situacao_candidado'] for item in x])
    df = df.explode(['codigo_list', 'situacao_candidato'])
    df = df.rename(columns={'codigo_list': 'applicant_id'})

    relevant_keywords = [
        'Encaminhado ao Requisitante', 'Contratado pela Decision', 'Documentação PJ', 'Aprovado',
        'Entrevista Técnica', 'Em avaliação pelo RH', 'Contratado como Hunting', 'Entrevista com Cliente',
        'Documentação CLT', 'Documentação Cooperado', 'Encaminhar Proposta', 'Proposta Aceita'
    ]
    hired_keywords = [
        'Contratado pela Decision', 'Aprovado', 'Contratado como Hunting', 'Encaminhar Proposta', 'Proposta Aceita'
    ]

    df['relevant'] = df['situacao_candidato'].apply(lambda x: 1 if x in relevant_keywords else 0)
    df['hired'] = df['situacao_candidato'].apply(lambda x: 1 if x in hired_keywords else 0)
    return df


def preprocess_applicants(df):
    """Preprocess applicants data."""
    columns_to_combine = [
        'titulo_profissional', 'objetivo_profissional', 'area_atuacao', 'conhecimentos_tecnicos',
        'certificacoes', 'nivel_profissional', 'nivel_academico', 'cursos', 'cv_pt',
        'nivel_ingles', 'nivel_espanhol'
    ]
    df = combine_columns(df, columns_to_combine)
    df = df.rename(columns={'applicants_id': 'applicant_id'})
    return df


def preprocess_jobs(df):
    """Preprocess jobs data."""
    columns_to_combine = [
        'titulo_vaga', 'nivel profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
        'areas_atuacao', 'principais_atividades', 'competencia_tecnicas_e_comportamentais',
        'habilidades_comportamentais_necessarias'
    ]
    df = combine_columns(df, columns_to_combine)
    df = df[['jobs_id', 'text']]
    return df


def calculate_similarities(df):
    """Calculate text similarities."""
    df['applicants_text'] = df['applicants_text'].fillna('')
    df['text'] = df['text'].fillna('')
    df['similarity'] = df.apply(lambda row: calculate_similarity(row['text'], row['applicants_text']), axis=1)
    return df


def group_data(df):
    """Group data by prospects."""
    return df.groupby('prospects_id').agg(
        applicant_id=("applicant_id", list),
        num_applicants=('applicant_id', 'count'),
        relevant=('relevant', list),
        hired=('hired', list),
        text=('text', 'first'),
        mean_similarity=('similarity', list),
    ).reset_index()


def initialize_recruiter_bot():
    """Initialize the RecruiterBot."""
    with open(os.path.join("src", "config", "index_config.yaml")) as f:
        index_cfg = yaml.safe_load(f)
    emb_mgr = EmbeddingManager(config_path=os.path.join("models_config.yaml"))
    indexer = FAISSIndexer(index_cfg)
    bot = RecruiterBot(emb_mgr, indexer)
    return bot, emb_mgr, indexer


def find_top_applicants(df, bot, emb_mgr, indexer):
    """Find top applicants for each job."""
    df['top_applicants'] = df.apply(
        lambda row: find_top_applicants_with_filters(
            job_description=row['text'],
            faiss_indexer=indexer,
            emb_mgr=emb_mgr,
            filters={},
            top_n=row['num_applicants']
        ),
        axis=1
    )
    df['top_applicants_ids'] = df['top_applicants'].apply(
        lambda x: [item['applicant_id'] for item in x if 'metadata' in item and 'jobs_id' in item['metadata']]
    )
    df['top_applicants_scores'] = df['top_applicants'].apply(
        lambda x: [item['score'] for item in x if 'score' in item]
    )
    return df


def calculate_validation_metrics(df):
    """Calculate validation metrics."""
    df['hired_similarity'] = df.apply(
        lambda row: [sim for hired, sim in zip(row['hired'], row['mean_similarity']) if hired == 1],
        axis=1
    )
    df['relevant_similarity'] = df.apply(
        lambda row: [sim for relevant, sim in zip(row['relevant'], row['mean_similarity']) if relevant == 1],
        axis=1
    )
    df['mean_prospects_similarity'] = df['mean_similarity'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
    df['mean_top_applicants_score'] = df['top_applicants_scores'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
    df['mean_hired_similarity'] = df['hired_similarity'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
    df['mean_relevant_similarity'] = df['relevant_similarity'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
    return df


def save_validation_data(df):
    """Save validation data to a file."""
    df = df[['prospects_id', 'applicant_id', 'top_applicants_ids', 'mean_prospects_similarity',
             'mean_top_applicants_score', 'mean_hired_similarity', 'mean_relevant_similarity']]
    df.to_parquet('data/processed/validation.parquet', index=False)


def print_summary(df):
    """Print summary metrics."""
    print(f"Mean hired similarity: {df['mean_hired_similarity'].mean():.4f}")
    print(f"Mean relevant similarity: {df['mean_relevant_similarity'].mean():.4f}")
    print(f"Mean prospects similarity: {df['mean_prospects_similarity'].mean():.4f}")
    print(f"Mean top applicants score: {df['mean_top_applicants_score'].mean():.4f}")


def main():
    df_prospects, df_applicants, df_jobs = load_data()

    df_prospects = preprocess_prospects(df_prospects)
    df_applicants = preprocess_applicants(df_applicants)
    df_jobs = preprocess_jobs(df_jobs)

    df_merged = df_prospects.merge(df_applicants[['applicant_id', 'text']], on='applicant_id', how='left').rename(columns={'text': 'applicants_text'})
    df_merged = df_merged.merge(df_jobs, left_on='prospects_id', right_on='jobs_id', how='left').dropna(subset=['applicant_id'])
    df_merged = df_merged.sample(frac=0.1, random_state=42)
    df_merged = calculate_similarities(df_merged)

    df_grouped = group_data(df_merged)

    bot, emb_mgr, indexer = initialize_recruiter_bot()
    df_grouped = find_top_applicants(df_grouped, bot, emb_mgr, indexer)

    df_validation = calculate_validation_metrics(df_grouped)
    save_validation_data(df_validation)
    print_summary(df_validation)


if __name__ == "__main__":
    main()