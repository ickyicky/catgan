import argparse
import yaml
from .utils import load_discriminator, load_generator
from .config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", metavar="FILE", help="configuration file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true", help="train models")
    group.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="generate cat image using pretrained model",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = Config.parse_obj(yaml.safe_load(f))

    discriminator = load_discriminator(config.discriminator.load_path)
    generator = load_generator(config.discriminator.load_path)

    if args.train:
        from .train import train_main

        train_main(generator, discriminator, config)
