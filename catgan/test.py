import torch
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from tqdm import tqdm
import wandb
import logging
from typing import Tuple, Optional
from .train import validate_step, configure as configure_train
from .wandb import log as wandblog
from .config import Config
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from .utils import transform, get_device
from .dataloader import CatsDataset


log = logging.getLogger()
log.setLevel(logging.DEBUG)


CONFIG: Optional[Config] = None


def configure(config: Config) -> None:
    """configure.

    :param config:
    :type config: Config
    :rtype: None
    """
    global CONFIG
    CONFIG = config
    configure_train(config)


def test(
    test_data: Tensor,
    generator: LSGANGenerator,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_criterion: torch.nn.MSELoss,
) -> None:
    """test.

    :param test_data:
    :type test_data: Tensor
    :param generator:
    :type generator: LSGANGenerator
    :param generator_criterion:
    :type generator_criterion: torch.nn.MSELoss
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param discriminator_criterion:
    :type discriminator_criterion: torch.nn.MSELoss
    :rtype: None
    """
    losses = {
        "test_d_real": [],
        "test_d_fake": [],
        "test_g": [],
        "test_d": [],
    }

    examples = {}

    with torch.no_grad():
        bar = tqdm(
            test_data,
            position=0,
            leave=False,
            desc=f"TEST",
        )
        last_batch_num = len(bar) - 1

        for i, batch in enumerate(bar):
            (loss_d_real, loss_d_fake, loss_g, fake, d_pred, g_pred,) = validate_step(
                generator,
                generator_criterion,
                discriminator,
                discriminator_criterion,
                batch,
            )

            losses["test_d_real"].append(loss_d_real)
            losses["test_d_fake"].append(loss_d_fake)
            losses["test_g"].append(loss_g)
            losses["test_d"].append(loss_d_real + loss_d_fake)

            # only log last validation batch to wandb, no need to spam it with images
            if i == last_batch_num:
                examples["fake"] = [
                    wandb.Image(img, caption=f"Pred: {val}")
                    for img, val in zip(fake.cpu(), g_pred.cpu())
                ]
                examples["real"] = [
                    wandb.Image(img, caption=f"Pred: {val}")
                    for img, val in zip(batch.cpu(), d_pred.cpu())
                ]

    avg_losses = {
        key: float(torch.Tensor(val).mean()) for key, val in losses.items() if val
    }
    log.info("Test results:")
    log.info(", ".join(f"{key}={val:.2f}" for key, val in avg_losses.items()))
    wandb_log = {
        "images": examples,
        "error": avg_losses,
    }
    wandblog(wandb_log)


def test_main(
    generator: LSGANGenerator,
    discriminator: LSGANDiscriminator,
    config: Config,
) -> None:
    """test_main.

    :param generator:
    :type generator: LSGANGenerator
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param config:
    :type config: Config
    :rtype: None
    """
    configure(config)

    dataset = CatsDataset(config.data.test_data, transform)
    data_loader = DataLoader(dataset, batch_size=config.data.batch_size)

    generator_criterion = torch.nn.MSELoss()
    discriminator_criterion = torch.nn.MSELoss()

    test(
        test_data=data_loader,
        generator=generator,
        generator_criterion=generator_criterion,
        discriminator=discriminator,
        discriminator_criterion=discriminator_criterion,
    )
