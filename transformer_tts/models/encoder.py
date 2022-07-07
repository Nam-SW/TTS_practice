import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import PositionalEncoding


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
        self.blocks.append(nn.Linear(embedding_size, hidden_size))

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512,
        max_position_embedding: int = 512,
        kernel_size: int = 512,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffnn_size: int = 1024,
        rate: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_position_embedding = max_position_embedding
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size
        self.rate = rate

        self.embedding = nn.Embedding(vocab_size, embedding_size, 0)
        self.prenet = EncoderPreNet(embedding_size, kernel_size, hidden_size, rate)
        self.postional_encoding = PositionalEncoding(
            hidden_size, rate, max_position_embedding
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, num_heads, ffnn_size, rate, batch_first=True
            ),
            num_layers,
        )

    def forward(self, input_ids, attention_mask=None):
        embedding = self.embedding(input_ids)
        prenet_output = self.prenet(embedding)
        prenet_output = self.postional_encoding(prenet_output)

        if attention_mask is None:
            attention_mask = torch.ne(input_ids, 0).to(input_ids.device)

        output = self.transformer_encoder(prenet_output, attention_mask)

        return output
