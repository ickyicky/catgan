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
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(True)

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.conv(x)
        out = self.batch_normalization(out)
        out = self.activation(out)
        return out


class LSGANDiscriminatorFullyConnectedBlock(nn.Module):
    def __init__(self, in_shape: Tuple[int, int], out_features: int):
        """Discriminator fully connected block

        :param in_shape:
        :type in_shape: Tuple[int]
        :param out_features:
        :type out_features: int
        """
        super().__init__()

        self.reshape = lambda x: x.view(x.shape[0], in_shape[0] * in_shape[1])
        self.linear = nn.Linear(
            in_features=in_shape[0] * in_shape[1],
            out_features=out_features,
            bias=False,
        )

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.reshape(x)
        out = self.linear(out)
        return out


class LSGANDiscriminator(nn.Module):
    def __init__(self):
        """Discriminator network.

        Expects (X, 3, 64, 64) tensors, where X is batch size.
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                kernel_size=5,
                in_channels=3,
                out_channels=64,
                stride=2,
                padding=4,
            ),
            nn.LeakyReLU(True),
        )

        self.conv2 = LSGANDiscriminatorConvBlock(
            kernel_size=5,
            in_channels=64,
            out_channels=128,
            stride=2,
            padding=2,
        )

        self.conv3 = LSGANDiscriminatorConvBlock(
            kernel_size=5,
            in_channels=128,
            out_channels=256,
            stride=2,
            padding=1,
        )

        self.conv4 = LSGANDiscriminatorConvBlock(
            kernel_size=5,
            in_channels=256,
            out_channels=512,
            stride=2,
        )

        self.fully_connected = LSGANDiscriminatorFullyConnectedBlock(
            in_shape=(512, 4),
            out_features=1,
        )

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.fully_connected(out)
        return out


if __name__ == "__main__":
    net = LSGANDiscriminator()
    noise = torch.randn(1, 3, 64, 64)
    out = net(noise)
    print(out.shape)
