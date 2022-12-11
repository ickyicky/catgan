import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import wandb
import logging
from .train import validate_model, batch_of_noise
from .wandb import log as wandblog
from .config import Config
from .networks.generator import LSGANGenerator
from .networks.discriminator import LSGANDiscriminator
from .utils import transform, get_device
from .dataloader import CatsDataset


log = logging.getLogger()
log.setLevel(logging.DEBUG)


def test_step(
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
    total_count = 3 * b_size
    correct = 0

    labels = torch.full((b_size,), real_label, dtype=torch.float)

    # train discriminator on real data
    loss_d_real, pred = validate_model(
        discriminator,
        batch,
        labels,
        discriminator_criterion,
        device,
    )
    _, actual_pred = torch.max(pred.data, 0)
    _, actual_labels = torch.max(labels.data, 0)
    correct += (actual_pred == actual_labels).sum().item()

    # train discriminator on fake data
    labels.fill_(fake_label)
    fake_batch = generator(batch_of_noise(b_size, device))
    loss_d_fake, pred = validate_model(
        discriminator,
        fake_batch.detach(),
        labels,
        discriminator_criterion,
        device,
    )
    _, actual_pred = torch.max(pred.data, 0)
    _, actual_labels = torch.max(labels.data, 0)
    correct += (actual_pred == actual_labels).sum().item()

    # train generator with trained discriminator
    labels.fill_(generator_fake_label)
    loss_g, pred = validate_model(
        discriminator,
        fake_batch,
        labels,
        generator_criterion,
        device,
    )
    _, actual_pred = torch.max(pred.data, 0)
    _, actual_labels = torch.max(labels.data, 0)
    correct += (actual_pred == actual_labels).sum().item()

    return loss_d_real, loss_d_fake, loss_g, fake_batch, pred, total_count, correct


def test(
    test_data,
    generator,
    generator_criterion,
    discriminator,
    discriminator_criterion,
    device,
    real_label,
    fake_label,
    generator_fake_label,
):
    losses = {
        "test_d_real": [],
        "test_d_fake": [],
        "test_g": [],
    }

    total = 0
    correct = 0

    with torch.no_grad():
        bar = tqdm(
            test_data,
            position=0,
            leave=False,
            desc=f"TEST",
        )
        last_batch_num = len(bar) - 1

        for i, batch in enumerate(bar):
            (
                loss_d_real,
                loss_d_fake,
                loss_g,
                fake,
                pred,
                total_count,
                correct,
            ) = test_step(
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

            losses["test_d_real"].append(loss_d_real)
            losses["test_d_fake"].append(loss_d_fake)
            losses["test_g"].append(loss_g)

            total += total_count
            correct += correct

            # only log last validation batch to wandb, no need to spam it with images
            if i == last_batch_num:
                cpu = torch.device("cpu")
                fake = fake.to(cpu)
                pred = pred.to(cpu)
                examples = [
                    wandb.Image(img, caption="Pred: {val}")
                    for img, val in zip(fake, pred)
                ]

    avg_losses = {key: float(torch.stack(val).mean()) for key, val in losses.items()}
    accuracy = correct / total
    log.info("Test results:")
    log.info(f"accuracy: {accuracy * 100}%")
    log.info(", ".join(f"{key}={val:.2f}" for key, val in avg_losses.items()))
    wandb_log = {
        "images": examples,
        "error": avg_losses,
        "accuracy": accuracy,
    }
    wandblog(wandb_log)


def test_main(
    generator: LSGANGenerator,
    discriminator: LSGANDiscriminator,
    config: Config,
):
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
        device=get_device(),
        real_label=config.real_label,
        fake_label=config.fake_label,
        generator_fake_label=config.generator_fake_label,
    )
