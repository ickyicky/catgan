from pydantic import BaseModel


class DataConfig(BaseModel):
    """DataConfig."""

    train_data: str
    test_data: str
    batch_size: int


class TrainConfig(BaseModel):
    """TrainConfig."""

    dis_learning_rate: float
    gen_learning_rate: float
    num_of_epochs: int
    steps_per_epoch: int


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

    real_label: float
    fake_label: float
    generator_fake_label: float
