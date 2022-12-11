import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim
from tqdm import tqdm
from typing import Any
import logging
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from .utils import transform, get_device
from .dataloader import CatsDataset
from .config import Config


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def batch_of_noise(b_size: int, device):
    """batch_of_noise.

    :param b_size:
    :type b_size: int
    :param device:
    """
    return torch.randn(b_size, 1024, device=device)


def common_compute(model, batch, device):
    """common_compute.

    :param model:
    :param batch:
    :param device:
    """
    batch = batch.to(device)
    result = model(batch).view(-1)
    return result


def calculate_loss(result, label, criterion, device):
    """calculate_loss.

    :param result:
    :param label:
    :param criterion:
    :param device:
    """
    label = label.to(device)
    return criterion(result, label)


def train_model(model, batch, label, criterion, device):
    """train_batch.

    :param model:
    :param batch:
    :param label:
    :param criterion:
    :param device:
    """
    result = common_compute(model, batch, device)
    loss = calculate_loss(result, label, criterion, device)
    loss.backward()
    return loss


def validate_model(model, batch, label, criterion, device):
    """train_batch.

    :param model:
    :param batch:
    :param label:
    :param criterion:
    :param device:
    """
    result = common_compute(model, batch, device)
    loss = calculate_loss(result, label, criterion, device)
    return loss


def train_step(
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
):
    """train_step.

    :param generator:
    :param generator_optimizer:
    :param generator_criterion:
    :param discriminator:
    :param discriminator_optimizer:
    :param discriminator_criterion:
    :param batch:
    :param device:
    :param real_label:
    :param fake_label:
    :param generator_fake_label:
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
    generator,
    generator_criterion,
    discriminator,
    discriminator_criterion,
    batch,
    device,
    real_label,
    fake_label,
    generator_fake_label,
):
    # create labels
    b_size = batch.size(0)
    labels = torch.full((b_size,), real_label, dtype=torch.float)

    # train discriminator on real data
    loss_d_real = validate_model(
        discriminator,
        batch,
        labels,
        discriminator_criterion,
        device,
    )

    # train discriminator on fake data
    labels.fill_(fake_label)
    fake_batch = generator(batch_of_noise(b_size, device))
    loss_d_fake = validate_model(
        discriminator,
        fake_batch.detach(),
        labels,
        discriminator_criterion,
        device,
    )

    # train generator with trained discriminator
    labels.fill_(generator_fake_label)
    loss_g = validate_model(
        discriminator,
        fake_batch,
        labels,
        generator_criterion,
        device,
    )

    return loss_d_real, loss_d_fake, loss_g


def train(
    train_data,
    validate_data,
    num_of_epochs,
    generator,
    generator_optimizer,
    generator_criterion,
    discriminator,
    discriminator_optimizer,
    discriminator_criterion,
    device,
    real_label,
    fake_label,
    generator_fake_label,
):
    for epoch in range(num_of_epochs):
        generator.train()
        discriminator.train()

        losses = {"d_real": [], "d_fake": [], "g": []}

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

            losses["d_real"].append(loss_d_real)
            losses["d_fake"].append(loss_d_fake)
            losses["g"].append(loss_g)

        logging_communicate = f"Epoch {epoch}. train losses: "
        for name, val in losses.items():
            avg_loss = torch.stack(val).mean()
            logging_communicate += f"{name}: {avg_loss:0.2f} "

        generator.eval()
        discriminator.eval()

        losses = {"d_real": [], "d_fake": [], "g": []}

        with torch.no_grad():
            for batch in tqdm(
                validate_data,
                position=0,
                leave=False,
                desc=f"VALIDATE epoch: {epoch}/{num_of_epochs}",
            ):
                loss_d_real, loss_d_fake, loss_g = validate_step(
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

                losses["d_real"].append(loss_d_real)
                losses["d_fake"].append(loss_d_fake)
                losses["g"].append(loss_g)

        logging_communicate += f"; train losses: "
        for name, val in losses.items():
            avg_loss = torch.stack(val).mean()
            logging_communicate += f"{name}: {avg_loss:0.2f} "

        log.info(logging_communicate)


def train_main(
    generator: LSGANGenerator,
    discriminator: LSGANDiscriminator,
    config: Config,
):
    """train_main.

    :param generator:
    :type generator: LSGANGenerator
    :param discriminator:
    :type discriminator: LSGANDiscriminator
    :param config:
    :type config: Config
    """
    dataset = CatsDataset(config.data.root_dir, transform)
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
