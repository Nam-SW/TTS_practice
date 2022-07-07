import torch.nn as nn
from models.utils import ResidualStack, WConv1d, WConvTranspose1d
from numpy import iterable


class Generator(nn.Module):
    def __init__(
        self,
        mel_filter: int = 80,
        input_size: int = 512,
        stride: iterable = [8, 8, 2, 2],
        residual_kernel_size: int = 3,
        residual_dilation: iterable = [1, 3, 9],
        padding_mode: str = "reflect",
        leakyrelu_factor: float = 0.2,
    ):
        super(Generator, self).__init__()

        self.mel_filter = mel_filter
        self.input_size = input_size
        self.stride = stride
        self.residual_kernel_size = residual_kernel_size
        self.residual_dilation = residual_dilation
        self.padding_mode = padding_mode
        self.leakyrelu_factor = leakyrelu_factor

        self.layers = nn.ModuleList()
        self.layers.append(
            WConv1d(
                mel_filter,
                self.input_size,
                7,
                padding=3,
                padding_mode=padding_mode,
            )
        )
        for i, s in enumerate(self.stride):
            block = nn.Sequential(
                nn.LeakyReLU(leakyrelu_factor),
                WConvTranspose1d(
                    self.input_size // max(i * 2, 1),
                    self.input_size // ((i + 1) * 2),
                    s * 2,
                    stride=s,
                    padding=s // 2 + s % 2,
                ),
                ResidualStack(
                    self.input_size // ((i + 1) * 2),
                    dilation=residual_dilation,
                    padding_mode=padding_mode,
                    leakyrelu_factor=leakyrelu_factor,
                ),
            )
            self.layers.append(block)

        self.layers.append(
            nn.Sequential(
                nn.LeakyReLU(leakyrelu_factor),
                WConv1d(
                    self.input_size // (len(self.stride) * 2),
                    1,
                    7,
                    padding=3,
                    padding_mode=padding_mode,
                ),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        def condition(l):
            return isinstance(l, (ResidualStack, WConv1d, WConvTranspose1d))

        for layer in self.layers:
            if condition(layer):
                layer.remove_weight_norm()
            elif isinstance(layer, nn.Sequential):
                for l in layer:
                    if condition(l):
                        l.remove_weight_norm()
