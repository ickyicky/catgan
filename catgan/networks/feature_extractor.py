from .discriminator import LSGANDiscriminator
from torch import Tensor


class FeatureExtractor(LSGANDiscriminator):
    """FeatureExtractor."""

    def forward(self, x: Tensor) -> Tensor:
        """forward. Omits last convolution that maps
        extracted features to final verdict

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out
