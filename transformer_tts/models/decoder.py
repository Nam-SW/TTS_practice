import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import PositionalEncoding


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
        filter_size: int = 5,
        rate: float = 0.1,
    ):
        super(DecoderPostnet, self).__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(mel_size, mel_size, filter_size, padding="same"),
                    nn.Tanh(),
                    nn.BatchNorm1d(),
                )
                for _ in range(5)
            ]
        )

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class Dncoder(nn.Module):
    def __init__(
        self,
        mel_size: int = 80,
        max_position_embedding: int = 512,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffnn_size: int = 1024,
        filter_size: int = 5,
        rate: float = 0.1,
    ):
        super(Dncoder, self).__init__()

        self.mel_size = mel_size
        self.max_position_embedding = max_position_embedding
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size
        self.filter_size = filter_size
        self.rate = rate

        self.prenet = DecoderPrenet(mel_size, hidden_size, rate)
        self.postional_encoding = PositionalEncoding(
            hidden_size, rate, max_position_embedding
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_size, num_heads, ffnn_size, rate, batch_first=True
            ),
            num_layers,
        )
        self.mel_linear = nn.Linear(hidden_size, mel_size)
        self.postnet = DecoderPostnet(mel_size, filter_size)
        self.stop_linear = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_output,
        spectrogram,
        attention_mask,
        decoder_attention_mask,
    ):
        x = self.prenet(spectrogram)
        x = self.postional_encoding(x)

        self.transformer_decoder(
            x,
            encoder_output,
            decoder_attention_mask,
            attention_mask,
        )
