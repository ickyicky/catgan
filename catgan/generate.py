from .networks.generator import LSGANGenerator
from .train import batch_of_noise
import torchvision.transforms as T


def generate(generator: LSGANGenerator) -> None:
    """generate.

    :param generator:
    :type generator: LSGANGenerator
    :rtype: None
    """
    noise = batch_of_noise(1, generator.in_features)
    cat_image = generator(noise).cpu()
    T.ToPILImage(cat_image).show()
