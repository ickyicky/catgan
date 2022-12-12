import wandb
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Union, Tuple, Optional
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from .utils import transform, get_device
from .dataloader import CatsDataset
from .config import Config
from .wandb import log as wandblog


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


CONFIG: Optional[Config] = None


def configure(config: Config) -> None:
    """configure.

    :param config:
    :type config: Config
    :rtype: None
    """
    global CONFIG
    CONFIG = config


def batch_of_noise(b_size: int, in_features: int) -> Tensor:
    return torch.randn(b_size, in_features, 1, 1, device=get_device())


def common_compute(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
) -> Tensor:
    """common_compute.

    :param model:
    :type model: Union[LSGANDiscriminator, LSGANGenerator]
    :param batch:
    :type batch: Tensor
    :rtype: Tensor
    """
    batch = batch.to(get_device())
    result = model(batch).view(-1)
    return result


def calculate_loss(
    result: Tensor, label: Tensor, criterion: torch.nn.MSELoss
) -> Tensor:
    """calculate_loss.

    :param result:
    :type result: Tensor
    :param label:
    :type label: Tensor
    :param criterion:
    :type criterion: torch.nn.MSELoss
    :rtype: Tensor
    """
    label = label.to(get_device())
    return criterion(result, label)


def train_model(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
    label: Tensor,
    criterion: torch.nn.MSELoss,
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
    :rtype: Tensor
    """
    result = common_compute(model, batch)
    loss = calculate_loss(result, label, criterion)
    loss.backward()
    return loss


def validate_model(
    model: Union[LSGANDiscriminator, LSGANGenerator],
    batch: Tensor,
    label: Tensor,
    criterion: torch.nn.MSELoss,
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
    :rtype: Tuple[Tensor, Tensor]
    """
    result = common_compute(model, batch)
    loss = calculate_loss(result, label, criterion)
    return loss, result


def train_step(
    generator: LSGANGenerator,
    generator_optimizer: optim.Adam,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_optimizer: optim.Adam,
    discriminator_criterion: torch.nn.MSELoss,
    batch: Tensor,
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
    :rtype: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    """
    b_size = batch.size(0)
    labels = torch.full((b_size,), CONFIG.real_label, dtype=torch.float)
    loss_g = loss_d_real = loss_d_fake = None

    # train generator with trained discriminator
    generator_optimizer.zero_grad()
    fake_batch = generator(batch_of_noise(b_size, generator.in_features))
    labels.fill_(CONFIG.generator_fake_label)
    loss_g = train_model(
        model=discriminator,
        batch=fake_batch,
        label=labels,
        criterion=generator_criterion,
    )
    if loss_g > config.train.gen_min_loss:
        generator_optimizer.step()

    # train discriminator on real data
    discriminator_optimizer.zero_grad()
    loss_d_real = train_model(
        model=discriminator,
        batch=batch,
        label=labels,
        criterion=discriminator_criterion,
    )

    # train discriminator on fake data
    labels.fill_(CONFIG.fake_label)
    fake_batch = generator(batch_of_noise(b_size, generator.in_features))
    loss_d_fake = train_model(
        model=discriminator,
        batch=fake_batch.detach(),
        label=labels,
        criterion=discriminator_criterion,
    )
    if torch.stack([loss_d_real, loss_d_fake]).mean() > config.train.dis_min_loss:
        discriminator_optimizer.step()

    return loss_d_real, loss_d_fake, loss_g


def validate_step(
    generator: LSGANGenerator,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_criterion: torch.nn.MSELoss,
    batch: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    :rtype: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    """
    b_size = batch.size(0)
    labels = torch.full((b_size,), CONFIG.real_label, dtype=torch.float)
    fake_batch = generator(batch_of_noise(b_size, generator.in_features))

    # train discriminator on real data
    loss_d_real, _ = validate_model(
        model=discriminator,
        batch=batch,
        label=labels,
        criterion=discriminator_criterion,
    )

    # train discriminator on fake data
    labels.fill_(CONFIG.fake_label)
    loss_d_fake, d_pred = validate_model(
        model=discriminator,
        batch=fake_batch.detach(),
        label=labels,
        criterion=discriminator_criterion,
    )

    # train generator with trained discriminator
    labels.fill_(CONFIG.generator_fake_label)
    loss_g, g_pred = validate_model(
        model=discriminator,
        batch=fake_batch,
        label=labels,
        criterion=generator_criterion,
    )

    return loss_d_real, loss_d_fake, loss_g, fake_batch, d_pred, g_pred


def train(
    train_data: Tensor,
    validate_data: Tensor,
    generator: LSGANGenerator,
    generator_optimizer: optim.Adam,
    generator_criterion: torch.nn.BCELoss,
    discriminator: LSGANDiscriminator,
    discriminator_optimizer: optim.Adam,
    discriminator_criterion: torch.nn.BCELoss,
) -> None:
    """train.

    :param train_data:
    :type train_data: Tensor
    :param validate_data:
    :type validate_data: Tensor
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
    :rtype: None
    """
    generator.train()
    discriminator.train()

    batch_iterator = iter(train_data)

    epochs = tqdm(
        range(CONFIG.train.num_of_epochs),
        position=0,
        leave=False,
        desc="TRAIN",
    )
    for epoch in epochs:

        losses = {
            "train_d_real": [],
            "train_d_fake": [],
            "train_g": [],
            "valid_d_real": [],
            "valid_d_fake": [],
            "valid_g": [],
        }

        try:
            batch = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_data)
            batch = next(batch_iterator)

        loss_d_real, loss_d_fake, loss_g = train_step(
            generator,
            generator_optimizer,
            generator_criterion,
            discriminator,
            discriminator_optimizer,
            discriminator_criterion,
            batch,
        )

        losses["train_d_real"].append(loss_d_real)
        losses["train_d_fake"].append(loss_d_fake)
        losses["train_g"].append(loss_g)

        if epoch > 0 and epoch % CONFIG.train.epochs_between_val == 0:
            generator.eval()
            discriminator.eval()
            examples = {}

            with torch.no_grad():
                bar = tqdm(validate_data, position=0, leave=False, desc=f"VALIDATE")
                last_batch_num = len(bar) - 1

                for i, batch in enumerate(bar):
                    (
                        loss_d_real,
                        loss_d_fake,
                        loss_g,
                        fake,
                        d_pred,
                        g_pred,
                    ) = validate_step(
                        generator,
                        generator_criterion,
                        discriminator,
                        discriminator_criterion,
                        batch,
                    )

                    losses["valid_d_real"].append(loss_d_real)
                    losses["valid_d_fake"].append(loss_d_fake)
                    losses["valid_g"].append(loss_g)

                    # only log last validation batch to wandb, no need to spam it with images
                    if i == last_batch_num:
                        cpu = torch.device("cpu")
                        examples["fake"] = [
                            wandb.Image(img, caption=f"Pred: {val}")
                            for img, val in zip(fake.to(cpu)[:3], g_pred.to(cpu)[:3])
                        ]
                        examples["real"] = [
                            wandb.Image(img, caption=f"Pred: {val}")
                            for img, val in zip(batch.to(cpu)[:3], d_pred.to(cpu)[:3])
                        ]

            avg_losses = {
                key: float(torch.stack(val).mean())
                for key, val in losses.items()
                if val
            }

            log.info(f"Epoch: {epoch}/{CONFIG.train.num_of_epochs}")
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

            epochs.set_description(
                "TRAIN:"
                + ", ".join(f"{key}={val:.2f}" for key, val in avg_losses.items())
            )


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
    configure(config)

    dataset = CatsDataset(config.data.train_data, transform)
    test_dataset = CatsDataset(config.data.test_data, transform)

    train_data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_data_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=config.train.gen_learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config.train.dis_learning_rate,
        betas=(0.5, 0.999),
    )

    generator_criterion = torch.nn.MSELoss()
    discriminator_criterion = torch.nn.MSELoss()

    train(
        train_data=train_data_loader,
        validate_data=val_data_loader,
        generator=generator,
        generator_optimizer=generator_optimizer,
        generator_criterion=generator_criterion,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_criterion=discriminator_criterion,
    )
