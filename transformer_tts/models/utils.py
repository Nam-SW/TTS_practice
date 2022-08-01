from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init)
        )

    def forward(self, x):
        return self.linear_layer(x)


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
    ):
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = hidden_size // num_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.depth]))

        self.build_model()

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        logits = matmul_qk / self.scale.to(q.device)
        if mask is not None:
            logits = logits.masked_fill(mask, -float("inf"))
            # logits = logits.masked_fill(mask, -1e9)
        score = F.softmax(logits, dim=-1)
        output = torch.matmul(score, v)
        return output, score

    def build_model(self):
        self.wq = Linear(self.hidden_size, self.hidden_size)
        self.wk = Linear(self.hidden_size, self.hidden_size)
        self.wv = Linear(self.hidden_size, self.hidden_size)

        self.dense = Linear(self.hidden_size, self.hidden_size)

    def split_head(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return torch.permute(x, [0, 2, 1, 3])

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_head(self.wq(q), batch_size)
        k = self.split_head(self.wq(k), batch_size)
        v = self.split_head(self.wq(v), batch_size)

        result, score = self.scaled_dot_product_attention(q, k, v, mask)
        result = torch.permute(result, [0, 2, 1, 3]).reshape(
            batch_size, -1, self.hidden_size
        )

        output = self.dense(result)

        return output, score


class PositionalEncodingWithAlpha(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncodingWithAlpha, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.pow(
            10000, torch.arange(0, hidden_size, 2).float() / hidden_size
        )
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        # self.embedding = nn.Embedding.from_pretrained(pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # idx = torch.arange(0, x.size(1), device=x.device)[None, :]
        # pe = self.embedding(idx)
        x = x + self.alpha * self.pe[: x.size(1)].unsqueeze(0)
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
                    Conv1d(
                        embedding_size,
                        embedding_size,
                        kernel_size,
                        padding="same",
                        w_init="relu",
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
                    Linear(
                        mel_size if i == 0 else hidden_size,
                        hidden_size,
                        w_init="relu",
                    ),
                    nn.ReLU(),
                    nn.Dropout(rate),
                )
            )
        self.norm = Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.norm(x)
        return x


class DecoderPostnet(nn.Module):
    def __init__(
        self,
        mel_size: int = 80,
        hidden_size: int = 256,
        kernel_size: int = 5,
    ):
        super(DecoderPostnet, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                Conv1d(
                    mel_size,
                    hidden_size,
                    kernel_size,
                    padding="same",
                    w_init="tanh",
                ),
                nn.Tanh(),
                nn.BatchNorm1d(hidden_size),
            )
        )
        for _ in range(3):
            self.blocks.append(
                nn.Sequential(
                    Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size,
                        padding="same",
                        w_init="tanh",
                    ),
                    nn.Tanh(),
                    nn.BatchNorm1d(hidden_size),
                )
            )
        self.blocks.append(Conv1d(hidden_size, mel_size, kernel_size, padding="same"))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.blocks:
            x = layer(x)
        return x.permute(0, 2, 1)
