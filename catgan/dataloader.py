from PIL import Image
from glob import glob

from torch.utils.data.dataset import Dataset
from torch import Tensor
from torchvision.transforms import Compose


class CatsDataset(Dataset):
    """Cats dataset.

    Basically https://github.com/gmalivenko/cat-gan/blob/master/dataset.py
    with minor changes, because its implementation was so plain and simple
    I was't able to write anything more elegant :)
    """

    def __init__(self, root_dir: str, transform: Compose):
        """__init__.

        :param root_dir:
        :type root_dir: str
        :param transform:
        :type transform: Compose
        """
        self.images = glob(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tensor:
        """__getitem__.

        :param idx:
        :type idx: int
        :rtype: Tensor
        """
        image = Image.open(self.images[idx])
        sample = self.transform(image)
        return sample
