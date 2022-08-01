import torch
import torch.nn as nn
from models.utils import Linear


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ffnn_size: int = 1024,
        rate: float = 0.2,
        norm_first: bool = False,
    ):
        super(DecoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ffnn_size = ffnn_size
        self.rate = rate
        self.norm_first = norm_first

        self.build_model()

    def build_model(self):
        self.mha1 = nn.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            dropout=self.rate,
            batch_first=True,
        )
        self.mha2 = nn.MultiheadAttention(
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
        self.norm3 = nn.LayerNorm(self.hidden_size)

        self.dropout1 = nn.Dropout(self.rate)
        self.dropout2 = nn.Dropout(self.rate)
        self.dropout3 = nn.Dropout(self.rate)

    def forward(self, x, e_output, mask=None, lh_mask=None, encoder_mask=None):
        if self.norm_first:
            x = self.norm1(x)
            attn, score1 = self.mha1(x, x, x, attn_mask=lh_mask, key_padding_mask=mask)
            x = self.dropout1(attn) + x

            x = self.norm1(x)
            attn, score2 = self.mha2(
                x, e_output, e_output, key_padding_mask=encoder_mask
            )
            x = self.dropout1(attn) + x

            x = self.norm2(x)
            ffnn = self.ffnn(x)
            output = self.dropout2(ffnn) + x

        else:
            attn, score1 = self.mha1(x, x, x, attn_mask=lh_mask, key_padding_mask=mask)
            x = self.norm1(self.dropout1(attn) + x)

            attn, score2 = self.mha2(
                x, e_output, e_output, key_padding_mask=encoder_mask
            )
            ffnn = self.ffnn(x)
            output = self.norm2(self.dropout2(ffnn) + x)

        return output, score1, score2


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffnn_size: int = 1024,
        rate: float = 0.2,
        norm_first: bool = False,
    ):
        super(Decoder, self).__init__()

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
                DecoderLayer(
                    self.hidden_size,
                    self.num_heads,
                    self.ffnn_size,
                    self.rate,
                    self.norm_first,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x, e_output, mask=None, lh_mask=None, encoder_mask=None):
        self_attn_scores = []
        encoder_attn_scores = []
        for layer in self.layers:
            x, score1, score2 = layer(x, e_output, mask, lh_mask, encoder_mask)
            self_attn_scores.append(score1.unsqueeze(0))
            encoder_attn_scores.append(score2.unsqueeze(0))

        return x, torch.cat(self_attn_scores), torch.cat(encoder_attn_scores)
