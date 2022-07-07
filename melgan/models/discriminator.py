import torch.nn as nn
from models.utils import WConv1d
from numpy import iterable


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        output_size: iterable = [16, 64, 256, 1024, 1024, 1024],
        kernel_size: iterable = [15, 41, 41, 41, 41, 5],
        stride: iterable = [1, 4, 4, 4, 4, 1],
        groups: iterable = [1, 4, 16, 64, 256, 1],
        padding_mode: str = "reflect",
        leakyrelu_factor: float = 0.2,
    ):
        super(DiscriminatorBlock, self).__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding_mode = padding_mode
        self.leakyrelu_factor = leakyrelu_factor

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    WConv1d(
                        1 if i == 0 else output_size[i - 1],
                        output_size[i],
                        kernel_size[i],
                        stride[i],
                        kernel_size[i] // 2,
                        groups=groups[i],
                        padding_mode=padding_mode,
                    ),
                    nn.LeakyReLU(leakyrelu_factor),
                )
                for i in range(len(output_size))
            ]
            + [
                WConv1d(
                    output_size[-1], 1, 3, padding=2 // 2, padding_mode=padding_mode
                )
            ]
        )

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs[-1], outputs[:-1]


class Discriminator(nn.Module):
    def __init__(
        self,
        output_size=[16, 64, 256, 1024, 1024, 1024],
        kernel_size=[15, 41, 41, 41, 41, 5],
        stride=[1, 4, 4, 4, 4, 1],
        groups=[1, 4, 16, 64, 256, 1],
        padding_mode="reflect",
        leakyrelu_factor: float = 0.2,
    ):
        super(Discriminator, self).__init__()

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding_mode = padding_mode
        self.leakyrelu_factor = leakyrelu_factor

        self.blocks = nn.ModuleList(
            [
                DiscriminatorBlock(
                    output_size,
                    kernel_size,
                    stride,
                    groups,
                    padding_mode,
                    leakyrelu_factor,
                )
                for _ in range(3)
            ]
        )
        self.avg = nn.AvgPool1d(4, 2, 1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        features = []

        for block in self.blocks:
            o, f = block(x)
            x = self.avg(x)

            outputs.append(o)
            features.append(f)

        return outputs, features
