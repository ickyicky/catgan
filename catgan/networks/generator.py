import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Union


class LSGANGeneratorDeconvBlock(nn.Module):
    """LSGANGeneratorDeconvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_normalization: bool = True,
        activation: bool = True,
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
        :param batch_normalization:
        :type batch_normalization: bool
        :param activation:
        :type activation: bool
        """
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
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

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.deconv(x)

        if self.has_bn:
            out = self.batch_normalization(out)

        if self.has_activation:
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

        self.reshape1 = lambda x: x.view(x.shape[0], x.shape[1])
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_shape[0] * out_shape[1] * out_shape[1],
            bias=False,
        )
        self.reshape2 = lambda x: x.view(
            x.shape[0], out_shape[0], out_shape[1], out_shape[1]
        )
        self.batch_normalization = nn.BatchNorm2d(out_shape[0])
        self.activation = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.reshape1(x)
        out = self.linear(out)
        out = self.reshape2(out)
        out = self.batch_normalization(out)
        out = self.activation(out)
        return out


class LSGANGenerator(nn.Module):
    def __init__(self, in_features: int):
        """Generator network.

        Excpects (X, 100, 1, 1) tensors, where X is batch size
        Outputs (X, 3, 64, 64) batch of generated images.
        """
        super().__init__()
        self.in_features = in_features

        # self.fully_connected = LSGANGeneratorFullyConnectedBlock(
        #     in_features=in_features,
        #     out_shape=(1024, 4),
        # )

        self.deconv1 = LSGANGeneratorDeconvBlock(
            kernel_size=4,
            in_channels=in_features,
            out_channels=1024,
            stride=1,
            padding=0,
        )

        self.deconv2 = LSGANGeneratorDeconvBlock(
            kernel_size=4,
            in_channels=1024,
            out_channels=512,
            stride=2,
            padding=1,
        )

        self.deconv3 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=512,
            out_channels=512,
            stride=1,
            padding=1,
        )

        self.deconv4 = LSGANGeneratorDeconvBlock(
            kernel_size=4,
            in_channels=512,
            out_channels=256,
            stride=2,
            padding=1,
        )

        self.deconv5 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=256,
            out_channels=256,
            stride=1,
            padding=1,
        )

        self.deconv6 = LSGANGeneratorDeconvBlock(
            kernel_size=4,
            in_channels=256,
            out_channels=128,
            stride=2,
            padding=1,
        )

        self.deconv7 = LSGANGeneratorDeconvBlock(
            kernel_size=3,
            in_channels=128,
            out_channels=128,
            stride=1,
            padding=1,
        )

        self.deconv8 = LSGANGeneratorDeconvBlock(
            kernel_size=4,
            in_channels=128,
            out_channels=3,
            stride=2,
            padding=1,
            activation=False,
            batch_normalization=False,
        )

        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)
        out = self.deconv8(out)
        out = self.tanh(out)
        return out


if __name__ == "__main__":
    net = LSGANGenerator(100)
    noise = torch.randn(1, 100, 1, 1)
    out = net(noise)
    print(out.shape)
