from pydantic import BaseModel


class DataConfig(BaseModel):
    train_data: str
    test_data: str
    batch_size: int


class TrainConfig(BaseModel):
    learning_rate: float
    num_of_epochs: int
    val_data_percentage: float


class ModelConfig(BaseModel):
    load_path: str
    save_to: str


class GeneratorConfig(ModelConfig):
    pass


class DiscriminatorConfig(ModelConfig):
    pass


class Config(BaseModel):
    data: DataConfig
    train: TrainConfig
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig

    real_label: int
    fake_label: int
    generator_fake_label: int
