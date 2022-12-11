import wandb
from typing import Any


def init(name: str):
    wandb.init(project=name)


def log(what: Any) -> None:
    if wandb.run is not None:
        wandb.log(what)
