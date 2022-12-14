from .networks.generator import LSGANGenerator
from .train import batch_of_noise
from math import sqrt, ceil
import torchvision.transforms as T
import matplotlib.pyplot as plt
from typing import List, Any


def generate_cat_images(generator: LSGANGenerator, amount: int) -> List[Any]:
    """generate.

    :param generator:
    :type generator: LSGANGenerator
    :param amount:
    :type amount: int
    :rtype: List[Any]
    """
    noise = batch_of_noise(amount, generator.in_features)
    cat_images = generator(noise).cpu()
    transform = T.ToPILImage()
    cat_images = [transform(cat_image).convert("RGB") for cat_image in cat_images]
    return cat_images


def generate(generator: LSGANGenerator, amount: int) -> None:
    """generate.

    :param generator:
    :type generator: LSGANGenerator
    :param amount:
    :type amount: int
    :rtype: None
    """
    cat_images = generate_cat_images(generator, amount)
    rows = int(sqrt(amount))
    cols = ceil(amount / rows)
    _, ax = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            try:
                ax[row][col].imshow(cat_images[cols * row + col])
            except IndexError:
                break

    plt.show()
