import os
import sys
import logging
import wandb
import torch
import torchvision.transforms as transforms
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from typing import Optional


log = logging.getLogger(__name__)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def set_logging(root):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if wandb.run is not None:
        if not os.path.exists("log"):
            os.mkdir("log")

        run_name = "_".join(wandb.run.name.split("-")[:-1])
        run_num = wandb.run.name.split("-")[-1]
        logfile = f"{run_num}_{run_name}.log"

        handler = logging.FileHandler(os.path.join("log", logfile))

        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)


def get_device():
    """get_device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def weights_init(model):
    classname = model.__class__.__name__
    if "LSGAN" in classname:
        pass
    elif "Conv" in classname:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)


def load_generator(load_path: Optional[str]) -> LSGANGenerator:
    model = LSGANGenerator()
    model.apply(weights_init)

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
    model.apply(weights_init)

    if load_path is not None:
        if not os.path.exists(load_path):
            log.warn(
                f"Given discriminator load_path: {load_path} does not exist, initiallizing new model"
            )
        else:
            model.load_state_dict(load_path)

    model = model.to(get_device())

    return model


def save_model(model, path: str) -> None:
    folder = os.path.join(*os.path.split(path)[:-1])
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)
