from pydantic import BaseModel


class DataConfig(BaseModel):
    """DataConfig."""

    train_data: str
    test_data: str
    batch_size: int


class TrainConfig(BaseModel):
    """TrainConfig."""

    learning_rate: float
    num_of_epochs: int
    val_data_percentage: float


class ModelConfig(BaseModel):
    """ModelConfig."""

    load_path: str
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

    real_label: int
    fake_label: int
    generator_fake_label: int
