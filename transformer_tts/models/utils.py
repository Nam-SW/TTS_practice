from math import log

import torch
import torch.nn as nn


class PositionalEncodingWithAlpha(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncodingWithAlpha, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.alpha * self.pe[: x.size(0)]
        return self.dropout(x)


class EncoderPreNet(nn.Module):
    def __init__(
        self,
        embedding_size: int = 512,
        kernel_size: int = 5,
        hidden_size: int = 256,
        rate: float = 0.1,
    ):
        super(EncoderPreNet, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        embedding_size,
                        embedding_size,
                        kernel_size,
                        padding="same",
                    ),
                    nn.BatchNorm1d(embedding_size),
                    nn.ReLU(),
                    nn.Dropout(rate),
                )
            )
        self.projection = nn.Linear(embedding_size, hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.blocks:
            x = layer(x)

        output = self.projection(x.permute(0, 2, 1))
        return output


class DecoderPrenet(nn.Module):
    def __init__(
        self,
        mel_size: int = 80,
        hidden_size: int = 256,
        rate: float = 0.1,
    ):
        super(DecoderPrenet, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(mel_size if i == 0 else hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(rate),
                )
            )

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class DecoderPostnet(nn.Module):
    def __init__(
        self,
        mel_size: int = 80,
        kernel_size: int = 5,
    ):
        super(DecoderPostnet, self).__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(mel_size, mel_size, kernel_size, padding="same"),
                    nn.Tanh(),
                    nn.BatchNorm1d(mel_size),
                )
                for _ in range(5)
            ]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.blocks:
            x = layer(x)
        return x.permute(0, 2, 1)
