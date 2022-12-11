from PIL import Image
from glob import glob

from torch.utils.data.dataset import Dataset


class CatsDataset(Dataset):
    """Cats dataset."""

    def __init__(self, root_dir, transform):
        """__init__.

        :param root_dir:
        :param transform:
        """
        self.images = glob(root_dir)
        self.transform = transform

    def __len__(self):
        """__len__."""
        return len(self.images)

    def __getitem__(self, idx):
        """__getitem__.

        :param idx:
        """
        image = Image.open(self.images[idx])
        sample = self.transform(image)
        return sample
