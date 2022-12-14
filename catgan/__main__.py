import argparse
import yaml
import logging
from .utils import load_discriminator, load_generator, set_logging, save_model
from .train import train_main
from .test import test_main
from .config import Config, override_config
from .generate import generate
from .wandb import init


log = logging.getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        help="configuration file",
        default="config.yaml",
    )
    parser.add_argument(
        "--amount",
        default=16,
        type=int,
        help="number of cat images to generate (only suitable for --generate)",
    )
    parser.add_argument(
        "CONFIGURATION",
        help="configuration overloading, for example: discriminator.load_path=None or real_label=2.0",
        nargs="*",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true", help="train models")
    group.add_argument(
        "-s",
        "--test",
        action="store_true",
        help="test model on test data (runs by default after train)",
    )
    group.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate cat image using pretrained model",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = override_config(config, args.CONFIGURATION)
    config = Config.parse_obj(config)
    set_logging(log, config.log_to_stdout)

    if args.train:
        init("catgan")

        discriminator = load_discriminator(config.discriminator.load_path)
        generator = load_generator(
            config.generator.load_path, config.generator.in_features
        )

        try:
            train_main(generator, discriminator, config)
        except KeyboardInterrupt:
            pass

        save_model(generator, config.generator.save_to)
        save_model(discriminator, config.discriminator.save_to)

    if args.test:

        discriminator = load_discriminator(config.discriminator.load_path)
        generator = load_generator(
            config.generator.load_path, config.generator.in_features
        )

        test_main(generator, discriminator, config)

    if args.generate:
        generator = load_generator(
            config.generator.load_path, config.generator.in_features
        )
        generate(generator, args.amount)
