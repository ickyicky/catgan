import os
import sys
import logging
import wandb
import torch
import torchvision.transforms as transforms
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from typing import Optional, Union


log = logging.getLogger(__name__)


"""
Basically https://github.com/gmalivenko/cat-gan.git
transformation, he did great job analyzing the dataset
"""
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.1,
            saturation=0.3,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def set_logging(root: logging.Logger) -> None:
    """set_logging.

    :param root:
    :type root: logging.Logger
    :rtype: None
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists("log"):
        os.mkdir("log")

    logfile = "run.log"

    if wandb.run is not None:
        run_name = "_".join(wandb.run.name.split("-")[:-1])
        run_num = wandb.run.name.split("-")[-1]
        logfile = f"{run_num}_{run_name}.log"

    handler = logging.FileHandler(os.path.join("log", logfile))

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root.addHandler(handler)


DEVICE: Optional[torch.device] = None


def get_device() -> torch.device:
    """get_device.

    :rtype: torch.device
    """
    global DEVICE

    if DEVICE is not None:
        return DEVICE

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return DEVICE


def weights_init(model: Union[LSGANDiscriminator, LSGANGenerator]) -> None:
    """weights_init.

    :param model:
    :type model: Union[LSGANDiscriminator, LSGANGenerator]
    :rtype: None
    """
    classname = model.__class__.__name__

    if "LSGAN" in classname:
        pass
    elif "Conv" in classname:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)


def load_generator(load_path: Optional[str], in_features: int) -> LSGANGenerator:
    """load_generator.

    :param load_path:
    :type load_path: Optional[str]
    :param in_features:
    :type in_features: int
    :rtype: LSGANGenerator
    """
    model = LSGANGenerator(in_features)

    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    else:
        model.apply(weights_init)

    model = model.to(get_device())

    return model


def load_discriminator(load_path: Optional[str]) -> LSGANDiscriminator:
    """load_discriminator.

    :param load_path:
    :type load_path: Optional[str]
    :rtype: LSGANDiscriminator
    """
    model = LSGANDiscriminator()

    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    else:
        model.apply(weights_init)

    model = model.to(get_device())

    return model


def save_model(model, path: str) -> None:
    """save_model.

    :param model:
    :param path:
    :type path: str
    :rtype: None
    """
    folder = os.path.join(*os.path.split(path)[:-1])
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)
