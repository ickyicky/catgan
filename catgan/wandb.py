import wandb
from typing import Any


def init(name: str):
    wandb.init(project=name)


def log(what: Any) -> None:
    wandb.log(what)
