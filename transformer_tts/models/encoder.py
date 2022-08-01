import torch
import torch.nn as nn
from models.utils import Linear


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffnn_size: int = 1024,
        rate: float = 0.2,
        norm_first: bool = False,
    ):
        super(EncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ffnn_size = ffnn_size
        self.rate = rate
        self.norm_first = norm_first

        self.build_model()

    def build_model(self):
        self.mha = nn.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            dropout=self.rate,
            batch_first=True,
        )
        self.ffnn = nn.Sequential(
            Linear(self.hidden_size, self.ffnn_size, w_init="relu"),
            nn.ReLU(),
            nn.Dropout(self.rate),
            Linear(self.ffnn_size, self.hidden_size),
        )

        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.dropout1 = nn.Dropout(self.rate)
        self.dropout2 = nn.Dropout(self.rate)

    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)
            attn, score = self.mha(x, x, x, key_padding_mask=mask)
            x = self.dropout1(attn) + x

            x = self.norm2(x)
            ffnn = self.ffnn(x)
            output = self.dropout2(ffnn) + x

        else:
            attn, score = self.mha(x, x, x, key_padding_mask=mask)
            x = self.norm1(self.dropout1(attn) + x)

            ffnn = self.ffnn(x)
            output = self.norm2(self.dropout2(ffnn) + x)

        return output, score


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffnn_size: int = 1024,
        rate: float = 0.2,
        norm_first: bool = False,
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size
        self.rate = rate
        self.norm_first = norm_first

        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.hidden_size,
                    self.num_heads,
                    self.ffnn_size,
                    self.rate,
                    self.norm_first,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, mask=None):
        scores = []
        for layer in self.layers:
            x, score = layer(x, mask)
            scores.append(score.unsqueeze(0))

        return x, torch.cat(scores)
