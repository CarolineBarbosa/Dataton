import json
import pandas as pd
from typing import Any, Dict
import yaml
import os
def load_yaml(file_path: str) -> Dict[str, Any]:

    """
    Load a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file
    :return: Dictionary containing the YAML file contents
    :raises FileNotFoundError: If the file does not exist
    :raises yaml.YAMLError: If the file contains invalid YAML
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file at {file_path}: {e}")

def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Load a Parquet file and return its contents as a pandas DataFrame.

    :param file_path: Path to the Parquet file
    :return: pandas DataFrame containing the Parquet file contents
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file cannot be read as a Parquet file
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise ValueError(f"Error reading Parquet file at {file_path}: {e}")
    
    
def load_json(file_info: Dict[str, str]) -> pd.DataFrame:
    """
    Load two JSON files and convert them to a pandas DataFrame.

    :param file_info: Dictionary with two keys: 'id' for the ID column name and 'file_path' for the JSON file path
    :return: Combined pandas DataFrame
    """
    rows = []
    id_column_name = file_info['id']
    json_file_path = file_info['path']

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item_id, item_info in data.items():
        row = {id_column_name: item_id}
        # Flatten nested information
        for key, value in item_info.items():
            if isinstance(value, dict):
                row.update(value)
            else:
                row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def save_to_parquet(df: pd.DataFrame, file_name: str) -> None:
    """Save a pandas DataFrame to a Parquet file."""
    prefix_path = "data/processed/"
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    output_file_path = f"{file_name}.parquet"
    df.to_parquet(prefix_path+output_file_path, index=False)

