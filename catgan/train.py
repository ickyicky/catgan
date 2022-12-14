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
from .networks.feature_extractor import FeatureExtractor
from .utils import transform, get_device
from .dataloader import CatsDataset
from .config import Config
from .wandb import log as wandblog
from .crosslid import compute_crosslid


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
    return (
        torch.FloatTensor(
            b_size,
            in_features,
            1,
            1,
        )
        .normal_(0, 1)
        .to(get_device())
    )


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
    return loss.item()


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
    return loss.item(), result


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

    # perform discriminator training on real data
    discriminator.zero_grad()
    labels = torch.full((b_size,), CONFIG.real_label, dtype=torch.float)
    loss_d_real = train_model(
        model=discriminator,
        batch=batch,
        label=labels,
        criterion=discriminator_criterion,
    )
    discriminator_optimizer.step()

    # perform discriminator training on fake data
    fake_batch = generator(batch_of_noise(b_size, generator.in_features))
    labels = torch.full((b_size,), CONFIG.fake_label, dtype=torch.float)
    loss_d_fake = train_model(
        model=discriminator,
        batch=fake_batch.detach(),
        label=labels,
        criterion=discriminator_criterion,
    )
    discriminator_optimizer.step()

    # train generator
    generator.zero_grad()
    fake_batch = generator(batch_of_noise(b_size, generator.in_features))
    labels = torch.full((b_size,), CONFIG.generator_fake_label, dtype=torch.float)
    loss_g = train_model(
        model=discriminator,
        batch=fake_batch,
        label=labels,
        criterion=generator_criterion,
    )

    feature_extractor = FeatureExtractor.from_discriminator(discriminator).to(
        get_device()
    )
    cross_lid = calculate_cross_lid(feature_extractor, fake_batch.detach(), batch)

    generator_optimizer.step()
    return loss_d_real, loss_d_fake, loss_g, cross_lid


def calculate_cross_lid(
    feature_extractor: FeatureExtractor,
    fake_batch: Tensor,
    real_batch: Tensor,
) -> Tensor:
    with torch.no_grad():
        b_size = real_batch.size(0)
        fake_features = feature_extractor(fake_batch.to(get_device())).cpu()
        real_features = feature_extractor(real_batch.to(get_device())).cpu()
        return compute_crosslid(
            fake_features,
            real_features,
            b_size,
            b_size,
        )


def validate_step(
    generator: LSGANGenerator,
    generator_criterion: torch.nn.MSELoss,
    discriminator: LSGANDiscriminator,
    discriminator_criterion: torch.nn.MSELoss,
    batch: Tensor,
    feature_extractor: FeatureExtractor,
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

    # validate discriminator on real data
    loss_d_real, d_pred = validate_model(
        model=discriminator,
        batch=batch,
        label=labels,
        criterion=discriminator_criterion,
    )

    # validate discriminator on fake data
    labels = torch.full((b_size,), CONFIG.fake_label, dtype=torch.float)
    loss_d_fake, _ = validate_model(
        model=discriminator,
        batch=fake_batch.detach(),
        label=labels,
        criterion=discriminator_criterion,
    )

    # validate generator with trained discriminator
    labels = torch.full((b_size,), CONFIG.generator_fake_label, dtype=torch.float)
    loss_g, g_pred = validate_model(
        model=discriminator,
        batch=fake_batch,
        label=labels,
        criterion=generator_criterion,
    )

    # calculate cross lid
    cross_lid = calculate_cross_lid(
        feature_extractor=feature_extractor,
        fake_batch=fake_batch,
        real_batch=batch,
    )

    return loss_d_real, loss_d_fake, loss_g, fake_batch, d_pred, g_pred, cross_lid


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
    batch_iterator = iter(train_data)

    epochs = tqdm(
        range(CONFIG.train.num_of_epochs),
        position=0,
        desc="TRAIN",
    )
    for epoch in epochs:
        generator.train()
        discriminator.train()

        losses = {
            "train_d_real": [],
            "train_d_fake": [],
            "train_d": [],
            "train_g": [],
            "train_cross_lid": [],
            "valid_d_real": [],
            "valid_d_fake": [],
            "valid_d": [],
            "valid_g": [],
            "valid_cross_lid": [],
        }

        for step in tqdm(
            range(CONFIG.train.steps_per_epoch), desc="STEP ", leave=False
        ):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_data)
                batch = next(batch_iterator)

            loss_d_real, loss_d_fake, loss_g, cross_lid = train_step(
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
            losses["train_d"].append(loss_d_real + loss_d_fake)
            losses["train_g"].append(loss_g)
            losses["train_cross_lid"].append(cross_lid)
            log.info(f"D: {loss_d_real + loss_d_fake} G: {loss_g}")
            wandblog(
                {
                    "epoch": epoch,
                    "step": step + epoch * CONFIG.train.steps_per_epoch,
                    "error": {
                        "train_d_real": loss_d_real,
                        "train_d_fake": loss_d_fake,
                        "train_d": loss_d_fake + loss_d_real,
                        "train_g": loss_g,
                        "train_cross_lid": cross_lid,
                    },
                }
            )

        generator.eval()
        discriminator.eval()
        examples = {}

        with torch.no_grad():
            bar = tqdm(validate_data, leave=False, desc="VALID")
            last_batch_num = len(bar) - 1
            feature_extractor = FeatureExtractor.from_discriminator(discriminator).to(
                get_device()
            )

            for i, batch in enumerate(bar):
                (
                    loss_d_real,
                    loss_d_fake,
                    loss_g,
                    fake,
                    d_pred,
                    g_pred,
                    cross_lid,
                ) = validate_step(
                    generator,
                    generator_criterion,
                    discriminator,
                    discriminator_criterion,
                    batch,
                    feature_extractor,
                )

                losses["valid_d_real"].append(loss_d_real)
                losses["valid_d_fake"].append(loss_d_fake)
                losses["valid_d"].append(loss_g + loss_d_real)
                losses["valid_g"].append(loss_g)
                losses["valid_cross_lid"].append(cross_lid)

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
        batch_size=config.data.val_batch_size,
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
