from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class TransformationConfig(BaseModel):
    """TransformationConfig."""

    brightness: float
    contrast: float
    saturation: float
    hue: float
    mean: List[float]
    std: List[float]
    size: int


class DataConfig(BaseModel):
    """DataConfig."""

    train_data: str
    val_data: str
    test_data: str
    batch_size: int
    val_batch_size: int
    transform: TransformationConfig


class TrainConfig(BaseModel):
    """TrainConfig."""

    dis_learning_rate: float
    gen_learning_rate: float
    num_of_epochs: int
    steps_per_epoch: int


class ModelConfig(BaseModel):
    """ModelConfig."""

    load_path: Optional[str]
    save_to: str


class GeneratorConfig(ModelConfig):
    """GeneratorConfig."""

    in_features: int


class DiscriminatorConfig(ModelConfig):
    """DiscriminatorConfig."""

    pass


class Config(BaseModel):
    """Config."""

    data: DataConfig
    train: TrainConfig
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig

    real_label: float
    fake_label: float
    generator_fake_label: float

    log_to_stdout: Optional[bool]


def override(config_dict: Dict[str, Any], key: str, val: str) -> Dict[str, Any]:
    """override.

    :param config_dict:
    :type config_dict: Dict[str, Any]
    :param key:
    :type key: str
    :param val:
    :type val: str
    :rtype: Dict[str, Any]
    """
    keys = key.split(".")
    accessed = config_dict

    if val == "None":
        val = None

    for key in keys[:-1]:
        accessed = accessed[key]

    accessed[keys[-1]] = val

    return config_dict


def override_config(config_dict: Dict[str, Any], from_cli: str) -> Dict[str, Any]:
    """override_config.

    :param config_dict:
    :type config_dict: Dict[str, Any]
    :param from_cli:
    :type from_cli: str
    :rtype: Dict[str, Any]
    """
    for value in from_cli:
        config_dict = override(config_dict, *value.split("="))

    return config_dict
