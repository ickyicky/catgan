from .networks.generator import LSGANGenerator
from .train import batch_of_noise
from math import sqrt, ceil
import torchvision.transforms as T
import matplotlib.pyplot as plt


def generate(generator: LSGANGenerator, amount: int) -> None:
    """generate.

    :param generator:
    :type generator: LSGANGenerator
    :param amount:
    :type amount: int
    :rtype: None
    """
    noise = batch_of_noise(amount, generator.in_features)
    cat_images = generator(noise).cpu()
    transform = T.ToPILImage()

    rows = int(sqrt(amount))
    cols = ceil(amount / rows)
    fig, ax = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            try:
                ax[row][col].imshow(
                    transform(cat_images[cols * row + col]).convert("RGB")
                )
            except IndexError:
                break

    plt.show()
