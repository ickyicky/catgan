import wandb
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Union, Tuple
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from .utils import transform, get_device
from .dataloader import CatsDataset
from .config import Config
from .wandb import log as wandblog


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def batch_of_noise(b_size: int, in_features: int, device: torch.device) -> Tensor:
    """batch_of_noise.

    :param b_size:
    :type b_size: int
    :param in_features:
    :type in_features: int
    :param device:
    :type device: torch.device
    :rtype: Tensor
    """
    return torch.randn(b_size, in_features, 1, 1, device=device)


def common_compute(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
    device: torch.device,
) -> Tensor:
    """common_compute.

    :param model:
    :type model: Union[LSGANDiscriminator, LSGANGenerator]
    :param batch:
    :type batch: Tensor
    :param device:
    :type device: torch.device
    :rtype: Tensor
    """
    batch = batch.to(device)
    result = model(batch).view(-1)
    return result


def calculate_loss(
    result: Tensor, label: Tensor, criterion: torch.nn.MSELoss, device: torch.device
) -> Tensor:
    """calculate_loss.

    :param result:
    :type result: Tensor
    :param label:
    :type label: Tensor
    :param criterion:
    :type criterion: torch.nn.MSELoss
    :param device:
    :type device: torch.device
    :rtype: Tensor
    """
    label = label.to(device)
    return criterion(result, label)


def train_model(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
    label: Tensor,
    criterion: torch.nn.MSELoss,
    device: torch.device,
) -> Tensor:
    """train_model.

    :param model:
    :type model: Union[LSGANDiscriminator, LSGANGenerator]
    :param batch:
    :type batch: Tensor
    :param label:
    :type label: Tensor
    :param criterion:
    :type criterion: torch.nn.MSELoss
    :param device:
    :type device: torch.device
    :rtype: Tensor
    """
    result = common_compute(model, batch, device)
    loss = calculate_loss(result, label, criterion, device)
    loss.backward()
    return loss


def validate_model(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
    label: Tensor,
    criterion: torch.nn.MSELoss,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """validate_model.

    :param model:
    :type model: Union[LSGANDiscriminator, LSGANGenerator]
    :param batch:
    :type batch: Tensor
    :param label:
    :type label: Tensor
    :param criterion:
    :type criterion: torch.nn.MSELoss
    :param device:
    :type device: torch.device
    :rtype: Tuple[Tensor, Tensor]
    """
    result = common_compute(model, batch, device)
    loss = calculate_loss(result, label, criterion, device)
    return loss, result


def train_step(
    generator: LSGANGenerator,
    generator_optimizer: optim.Adam,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_optimizer: optim.Adam,
    discriminator_criterion: torch.nn.MSELoss,
    batch: Tensor,
    device: torch.device,
    real_label: int,
    fake_label: int,
    generator_fake_label: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """train_step.

    :param generator:
    :type generator: LSGANGenerator
    :param generator_optimizer:
    :type generator_optimizer: optim.Adam
    :param generator_criterion:
    :type generator_criterion: torch.nn.MSELoss
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param discriminator_optimizer:
    :type discriminator_optimizer: optim.Adam
    :param discriminator_criterion:
    :type discriminator_criterion: torch.nn.MSELoss
    :param batch:
    :type batch: Tensor
    :param device:
    :type device: torch.device
    :param real_label:
    :type real_label: int
    :param fake_label:
    :type fake_label: int
    :param generator_fake_label:
    :type generator_fake_label: int
    :rtype: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    """
    # create labels
    b_size = batch.size(0)
    labels = torch.full((b_size,), real_label, dtype=torch.float)

    # train discriminator on real data
    discriminator_optimizer.zero_grad()
    loss_d_real = train_model(
        discriminator,
        batch,
        labels,
        discriminator_criterion,
        device,
    )

    # train discriminator on fake data
    labels.fill_(fake_label)
    fake_batch = generator(batch_of_noise(b_size, device))
    loss_d_fake = train_model(
        discriminator,
        fake_batch.detach(),
        labels,
        discriminator_criterion,
        device,
    )
    discriminator_optimizer.step()

    # train generator with trained discriminator
    generator_optimizer.zero_grad()
    labels.fill_(generator_fake_label)
    loss_g = train_model(
        discriminator,
        fake_batch,
        labels,
        generator_criterion,
        device,
    )
    generator_optimizer.step()

    return loss_d_real, loss_d_fake, loss_g


def validate_step(
    generator: LSGANGenerator,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_criterion: torch.nn.MSELoss,
    batch: Tensor,
    device: torch.device,
    real_label: int,
    fake_label: int,
    generator_fake_label: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """validate_step.

    :param generator:
    :type generator: LSGANGenerator
    :param generator_criterion:
    :type generator_criterion: torch.nn.MSELoss
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param discriminator_criterion:
    :type discriminator_criterion: torch.nn.MSELoss
    :param batch:
    :type batch: Tensor
    :param device:
    :type device: torch.device
    :param real_label:
    :type real_label: int
    :param fake_label:
    :type fake_label: int
    :param generator_fake_label:
    :type generator_fake_label: int
    :rtype: Tuple[Tensor, Tensor, Tensor, Tensor]
    """
    # create labels
    b_size = batch.size(0)
    labels = torch.full((b_size,), real_label, dtype=torch.float)

    # train discriminator on real data
    loss_d_real, _ = validate_model(
        discriminator,
        batch,
        labels,
        discriminator_criterion,
        device,
    )

    # train discriminator on fake data
    labels.fill_(fake_label)
    fake_batch = generator(batch_of_noise(b_size, generator.in_features, device))
    loss_d_fake, _ = validate_model(
        discriminator,
        fake_batch.detach(),
        labels,
        discriminator_criterion,
        device,
    )

    # train generator with trained discriminator
    labels.fill_(generator_fake_label)
    loss_g, pred = validate_model(
        discriminator,
        fake_batch,
        labels,
        generator_criterion,
        device,
    )

    return loss_d_real, loss_d_fake, loss_g, fake_batch, pred


def train(
    train_data: Tensor,
    validate_data: Tensor,
    num_of_epochs: int,
    generator: LSGANGenerator,
    generator_optimizer: optim.Adam,
    generator_criterion: torch.nn.BCELoss,
    discriminator: LSGANDiscriminator,
    discriminator_optimizer: optim.Adam,
    discriminator_criterion: torch.nn.BCELoss,
    device: torch.device,
    real_label: int,
    fake_label: int,
    generator_fake_label: int,
) -> None:
    """train.

    :param train_data:
    :type train_data: Tensor
    :param validate_data:
    :type validate_data: Tensor
    :param num_of_epochs:
    :type num_of_epochs: int
    :param generator:
    :type generator: LSGANGenerator
    :param generator_optimizer:
    :type generator_optimizer: optim.Adam
    :param generator_criterion:
    :type generator_criterion: torch.nn.BCELoss
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param discriminator_optimizer:
    :type discriminator_optimizer: optim.Adam
    :param discriminator_criterion:
    :type discriminator_criterion: torch.nn.BCELoss
    :param device:
    :type device: torch.device
    :param real_label:
    :type real_label: int
    :param fake_label:
    :type fake_label: int
    :param generator_fake_label:
    :type generator_fake_label: int
    :rtype: None
    """
    for epoch in range(num_of_epochs):
        generator.train()
        discriminator.train()

        losses = {
            "train_d_real": [],
            "train_d_fake": [],
            "train_g": [],
            "valid_d_real": [],
            "valid_d_fake": [],
            "valid_g": [],
        }

        for batch in tqdm(
            train_data,
            position=0,
            leave=False,
            desc=f"TRAIN epoch: {epoch}/{num_of_epochs}",
        ):
            loss_d_real, loss_d_fake, loss_g = train_step(
                generator,
                generator_optimizer,
                generator_criterion,
                discriminator,
                discriminator_optimizer,
                discriminator_criterion,
                batch,
                device,
                real_label,
                fake_label,
                generator_fake_label,
            )

            losses["train_d_real"].append(loss_d_real)
            losses["train_d_fake"].append(loss_d_fake)
            losses["train_g"].append(loss_g)

        generator.eval()
        discriminator.eval()
        examples = None

        with torch.no_grad():
            bar = tqdm(
                validate_data,
                position=0,
                leave=False,
                desc=f"VALIDATE epoch: {epoch}/{num_of_epochs}",
            )
            last_batch_num = len(bar) - 1

            for i, batch in enumerate(bar):
                loss_d_real, loss_d_fake, loss_g, fake, pred = validate_step(
                    generator,
                    generator_criterion,
                    discriminator,
                    discriminator_criterion,
                    batch,
                    device,
                    real_label,
                    fake_label,
                    generator_fake_label,
                )

                losses["valid_d_real"].append(loss_d_real)
                losses["valid_d_fake"].append(loss_d_fake)
                losses["valid_g"].append(loss_g)

                # only log last validation batch to wandb, no need to spam it with images
                if i == last_batch_num:
                    cpu = torch.device("cpu")
                    fake = fake.to(cpu)
                    pred = pred.to(cpu)
                    examples = [
                        wandb.Image(img, caption=f"Pred: {val}")
                        for img, val in zip(fake, pred)
                    ]

        avg_losses = {
            key: float(torch.stack(val).mean()) for key, val in losses.items()
        }
        log.info(f"Epoch: {epoch}/{num_of_epochs}")
        log.info(", ".join(f"{key}={val:.2f}" for key, val in avg_losses.items()))
        wandb_log = {
            "epoch": epoch,
            "learning_rate": {
                "generator": generator_optimizer.param_groups[0]["lr"],
                "discriminator": discriminator_optimizer.param_groups[0]["lr"],
            },
            "images": examples,
            "error": avg_losses,
        }
        wandblog(wandb_log)


def train_main(
    generator: LSGANGenerator,
    discriminator: LSGANDiscriminator,
    config: Config,
) -> None:
    """train_main.

    :param generator:
    :type generator: LSGANGenerator
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param config:
    :type config: Config
    :rtype: None
    """
    dataset = CatsDataset(config.data.train_data, transform)
    val_data_size = int(config.train.val_data_percentage * len(dataset))
    train_data, val_data = random_split(
        dataset, [len(dataset) - val_data_size, val_data_size]
    )
    train_data_loader = DataLoader(train_data, batch_size=config.data.batch_size)
    val_data_loader = DataLoader(val_data, batch_size=config.data.batch_size)

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=config.train.learning_rate
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=config.train.learning_rate
    )

    generator_criterion = torch.nn.MSELoss()
    discriminator_criterion = torch.nn.MSELoss()

    train(
        train_data=train_data_loader,
        validate_data=val_data_loader,
        num_of_epochs=config.train.num_of_epochs,
        generator=generator,
        generator_optimizer=generator_optimizer,
        generator_criterion=generator_criterion,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_criterion=discriminator_criterion,
        device=get_device(),
        real_label=config.real_label,
        fake_label=config.fake_label,
        generator_fake_label=config.generator_fake_label,
    )
