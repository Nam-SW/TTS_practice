import torch
import torch.nn as nn
from models.utils import (
    DecoderPostnet,
    DecoderPrenet,
    EncoderPreNet,
    PositionalEncodingWithAlpha,
)


class TransformerTTS(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512,
        max_position_embedding: int = 512,
        kernel_size: int = 5,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffnn_size: int = 1024,
        mel_size: int = 80,
        rate: float = 0.2,
    ):
        super(TransformerTTS, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_position_embedding = max_position_embedding
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size
        self.mel_size = mel_size
        self.rate = rate

        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, 0)
        self.encoder_prenet = EncoderPreNet(
            self.embedding_size, self.kernel_size, self.hidden_size, self.rate
        )
        self.decoder_prenet = DecoderPrenet(
            self.mel_size,
            self.hidden_size,
            self.rate,
        )
        self.postional_encoding = PositionalEncodingWithAlpha(
            self.hidden_size, self.rate, self.max_position_embedding
        )

        self.transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.ffnn_size,
            dropout=self.rate,
            batch_first=True,
            norm_first=False,
        )

        self.mel_linear = nn.Linear(self.hidden_size, self.mel_size)
        self.postnet = DecoderPostnet(self.mel_size, self.kernel_size)
        self.stop_linear = nn.Linear(self.hidden_size, 1)

    def create_padding_mask(self, inputs):
        return inputs == 0

    def create_decoder_mask(self, size):
        return torch.triu(torch.ones((size, size)) * float("-inf"), diagonal=1)

    def forward(
        self,
        input_ids,
        mel_spectrogram,
        attention_mask=None,
        decoder_attention_mask=None,
    ):
        if mel_spectrogram.size(-1) != self.mel_size:
            mel_spectrogram = mel_spectrogram.permute(0, 2, 1)

        embedding = self.embedding(input_ids)
        encoder_input = self.encoder_prenet(embedding)
        encoder_input = self.postional_encoding(encoder_input)

        decoder_input = self.decoder_prenet(mel_spectrogram)
        decoder_input = self.postional_encoding(decoder_input)

        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        if decoder_attention_mask is None:
            decoder_attention_mask = self.create_padding_mask(mel_spectrogram[:, :, 0])

        transformer_output = self.transformer(
            encoder_input,
            decoder_input,
            src_key_padding_mask=attention_mask,
            tgt_key_padding_mask=decoder_attention_mask,
            tgt_mask=self.create_decoder_mask(decoder_input.size(1)),
        )

        mel_output = self.mel_linear(transformer_output)
        mel_output += self.postnet(mel_output)
        stop_output = self.stop_linear(transformer_output)
        stop_output = torch.sigmoid(stop_output)

        return mel_output, stop_output
