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

    @classmethod
    def from_discriminator(
        self, discriminator: LSGANDiscriminator
    ) -> "FeatureExtractor":
        fe = FeatureExtractor()
        fe.load_state_dict(discriminator.state_dict())
        return fe


if __name__ == "__main__":
    import torch

    net = FeatureExtractor()
    noise = torch.randn(1, 3, 64, 64)
    out = net(noise)
    print(out.shape)
    from torchsummary import summary

    summary(net, (3, 64, 64))
