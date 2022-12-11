import wandb
from typing import Any, Dict


def init(name: str) -> None:
    """init.

    :param name:
    :type name: str
    :rtype: None
    """
    wandb.init(project=name)


def log(what: Dict[str, Any]) -> None:
    """log.

    :param what:
    :type what: Dict[str, Any]
    :rtype: None
    """
    if wandb.run is not None:
        wandb.log(what)
