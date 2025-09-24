import pandas as pd
from utils import load_json, save_to_parquet

def ingest_data(file_config) -> None:
    """
    Ingest data from a JSON file and convert it to a pandas DataFrame.

    :param file_path: Path to the JSON file
    :param id_column_name: Name of the ID column in the DataFrame
    :return: pandas DataFrame containing the ingested data
    """
    df = load_json(file_config)
    name = f"{file_config['path'].split('/')[-1].replace('.json', '')}"
    save_to_parquet(df, name)
    return 

def preprocessing(datasource_config) -> None:
    """
    Run the ingestion pipeline to convert JSON data to Parquet format for predefined datasources.

    :param input_paths: Dictionary mapping datasource names to their respective file paths
    """

    for entity, entity_config in datasource_config.items():
        print(f"Ingesting data for entity: {entity}")
        print(f"Path: {entity_config['path']}")
        ingest_data(entity_config)


