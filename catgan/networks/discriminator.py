import torch
import torch.nn as nn
from typing import Tuple


class LSGANDiscriminatorConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        batch_normalization: bool = True,
        activation: bool = True,
    ):
        """Discriminator conv block consists from a convolutional
        layer with parametrized kernel size, output filters and stride,
        which is followed by a batch normalization layer.

        :param in_channels:
        :type in_channels: int
        :param out_channels:
        :type out_channels: int
        :param kernel_size:
        :type kernel_size: int
        :param stride:
        :type stride: int
        :param padding:
        :type padding: int
        :param out_padding:
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.has_bn = batch_normalization
        if batch_normalization:
            self.batch_normalization = nn.BatchNorm2d(out_channels)

        self.has_activation = activation
        if self.has_activation:
            self.activation = nn.ReLU(True)

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.conv(x)

        if self.has_bn:
            out = self.batch_normalization(out)

        if self.has_activation:
            out = self.activation(out)

        return out


class LSGANDiscriminator(nn.Module):
    def __init__(self):
        """Discriminator network.

        Expects (X, 3, 64, 64) tensors, where X is batch size.
        """
        super().__init__()

        self.conv1 = LSGANDiscriminatorConvBlock(
            kernel_size=4,
            in_channels=3,
            out_channels=64,
            stride=2,
            padding=1,
            batch_normalization=False,
        )

        self.conv2 = LSGANDiscriminatorConvBlock(
            kernel_size=4,
            in_channels=64,
            out_channels=128,
            stride=2,
            padding=1,
        )

        self.conv3 = LSGANDiscriminatorConvBlock(
            kernel_size=4,
            in_channels=128,
            out_channels=256,
            stride=2,
            padding=1,
        )

        self.conv4 = LSGANDiscriminatorConvBlock(
            kernel_size=4,
            in_channels=256,
            out_channels=512,
            stride=2,
            padding=1,
        )

        self.conv5 = LSGANDiscriminatorConvBlock(
            kernel_size=4,
            in_channels=512,
            out_channels=1,
            stride=1,
            padding=0,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    net = LSGANDiscriminator()
    noise = torch.randn(1, 3, 64, 64)
    out = net(noise)
    print(out.shape)
