import torch
import torch.nn as nn
import torch.nn.functional as F


class WConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WConv1d, self).__init__()
        # self.layer = WeightNorm(nn.Conv1d(*args, **kwargs), ["weight", "bias"])
        self.layer = nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))
        self.layer.weight = self.layer.weight_v.detach()

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.layer)


class WConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WConvTranspose1d, self).__init__()
        # self.layer = WeightNorm(nn.ConvTranspose1d(*args, **kwargs), ["weight", "bias"])
        self.layer = nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))
        self.layer.weight = self.layer.weight_v.detach()

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.layer)


class ResidualStack(nn.Module):
    def __init__(
        self,
        input_size,
        kernel_size=3,
        dilation=[1, 3, 9],
        padding_mode="reflect",
        leakyrelu_factor=0.3,
    ):
        super(ResidualStack, self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.leakyrelu_factor = leakyrelu_factor

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(leakyrelu_factor),
                    WConv1d(
                        input_size,
                        input_size,
                        kernel_size,
                        padding=kernel_size // 2 + (d - 1),
                        padding_mode=padding_mode,
                        dilation=d,
                    ),
                    nn.LeakyReLU(leakyrelu_factor),
                    WConv1d(
                        input_size,
                        input_size,
                        kernel_size,
                        padding=kernel_size // 2,
                        padding_mode=padding_mode,
                    ),
                )
                for d in dilation
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = residual + x

        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            for l in layer:
                if isinstance(l, WConv1d):
                    l.remove_weight_norm()
