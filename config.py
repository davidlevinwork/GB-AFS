import yaml
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_path: str
    train_test_split: str
    label_column_str: str


@dataclass
class KFoldConfig:
    n_splits: int
    shuffle: bool


@dataclass
class TSNEConfig:
    n_iter: int
    perplexity: int
    n_components: int


@dataclass
class KMedoidsConfig:
    method: str


@dataclass
class Config:
    mode: str
    dataset: DatasetConfig
    k_fold: KFoldConfig
    t_sne: TSNEConfig
    k_medoids: KMedoidsConfig


def from_dict(data: Dict[str, Any]) -> Config:
    dataset = DatasetConfig(**data['dataset'])
    k_fold = KFoldConfig(**data['k_fold'])
    t_sne = TSNEConfig(**data['t_sne'])
    k_medoids = KMedoidsConfig(**data['k_medoids'])

    return Config(
        mode=data['mode'],
        dataset=dataset,
        k_fold=k_fold,
        t_sne=t_sne,
        k_medoids=k_medoids
    )


def load_yaml_config(file_path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config data class with parsed configuration values.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return from_dict(data)


# Load and parse the YAML configuration file
config = load_yaml_config('config.yaml')

