import os
import logging
import torch
import torchvision.transforms as transforms
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from typing import Optional


log = logging.getLogger(__name__)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def get_device():
    """get_device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_generator(load_path: Optional[str]) -> LSGANGenerator:
    model = LSGANGenerator()

    if load_path is not None:
        if not os.path.exists(load_path):
            log.warn(
                f"Given generator load_path: {load_path} does not exist, initiallizing new model"
            )
        else:
            model.load_state_dict(load_path)

    model = model.to(get_device())

    return model


def load_discriminator(load_path: Optional[str]) -> LSGANDiscriminator:
    model = LSGANDiscriminator()

    if load_path is not None:
        if not os.path.exists(load_path):
            log.warn(
                f"Given discriminator load_path: {load_path} does not exist, initiallizing new model"
            )
        else:
            model.load_state_dict(load_path)

    model = model.to(get_device())

    return model
