
from utils import load_yaml, load_json, save_to_parquet
class DatasourceConfig:
    def __init__(self, config_path: str):
        """
        Initialize the DatasourceConfig class.

        :param config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load the YAML configuration file.

        :return: Dictionary containing the configuration
        """
        return load_yaml(self.config_path)["datasources"]

    def get_datasource(self, name: str) -> dict:
        """
        Retrieve the configuration for a specific datasource.

        :param name: Name of the datasource
        :return: Dictionary containing the datasource configuration
        :raises KeyError: If the datasource is not found in the configuration
        """
        if name not in self.config:
            raise KeyError(f"Datasource '{name}' not found in configuration")
        return self.config[name]