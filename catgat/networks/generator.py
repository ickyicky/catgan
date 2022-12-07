import torch
import torch.nn as nn
from typing import Tuple


class LSGANGeneratorDeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        out_padding: int = 0,
    ):
        """Generator deconv block consists from a convolutional
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
        :type out_padding: int
        """
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
        )
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.deconv(x)
        out = self.batch_normalization(out)
        out = self.activation(out)
        return out


class LSGANGeneratorFullyConnectedBlock(nn.Module):
    def __init__(self, in_features: int, out_shape: Tuple[int, int]):
        """Generator fully connected block

        :param in_features:
        :type in_features: int
        :param out_shape:
        :type out_shape: Tuple[int, int]
        """
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_shape[0] * out_shape[1] * out_shape[1],
            bias=False,
        )
        self.reshape = lambda x: x.view(
            x.shape[0], out_shape[0], out_shape[1], out_shape[1]
        )
        self.batch_normalization = nn.BatchNorm2d(out_shape[0])
        self.activation = nn.ReLU(True)

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.linear(x)
        out = self.reshape(out)
        out = self.batch_normalization(out)
        out = self.activation(out)
        return out


class LSGANGenerator(nn.Module):
    def __init__(self):
        """Generator network.

        Excpects (X, 1024) tensors, where X is batch size
        Outputs (X, 3, 64, 64) batch of generated images.
        """
        super().__init__()

        self.fully_connected = LSGANGeneratorFullyConnectedBlock(
            in_features=1024,
            out_shape=(256, 7),
        )

        self.deconv1 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            stride=2,
            padding=2,
        )

        self.deconv2 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            stride=1,
            padding=1,
        )

        self.deconv3 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            stride=2,
            padding=2,
        )

        self.deconv4 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            stride=1,
            padding=1,
        )

        self.deconv5 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=128,
            stride=2,
            padding=3,
        )

        self.deconv6 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=128,
            out_channels=64,
            stride=2,
            padding=3,
        )

        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """forward.

        :param x:
        """
        out = self.fully_connected(x)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)
        return out


if __name__ == "__main__":
    net = LSGANGenerator()
    noise = torch.randn(1, 1024)
    out = net(noise)
    print(out.shape)
