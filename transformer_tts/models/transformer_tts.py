import torch
import torch.nn as nn
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import (
    DecoderPostnet,
    DecoderPrenet,
    EncoderPreNet,
    Linear,
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
        norm_first: bool = False,
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
        self.norm_first = norm_first

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

        self.encoder_pe = PositionalEncodingWithAlpha(
            self.hidden_size, self.rate, self.max_position_embedding
        )
        self.decoder_pe = PositionalEncodingWithAlpha(
            self.hidden_size, self.rate, self.max_position_embedding
        )

        self.encoder = Encoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffnn_size=self.ffnn_size,
            rate=self.rate,
            norm_first=self.norm_first,
        )
        self.decoder = Decoder(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffnn_size=self.ffnn_size,
            rate=self.rate,
            norm_first=self.norm_first,
        )

        self.mel_linear = Linear(self.hidden_size, self.mel_size)
        self.postnet = DecoderPostnet(
            self.mel_size,
            self.hidden_size,
            self.kernel_size,
        )
        self.stop_linear = Linear(self.hidden_size, 1, w_init="sigmoid")

    def get_padding_mask(self, inputs):
        return inputs == 0

    def get_decoder_mask(self, size):
        return torch.triu(torch.full((size, size), -float("inf")).float(), diagonal=1)
        # return torch.triu(torch.ones((size, size)).bool(), diagonal=1)

    def forward(
        self,
        input_ids,
        input_mel,
        attention_mask=None,
        decoder_attention_mask=None,
    ):

        if input_mel.size(-1) != self.mel_size:
            input_mel = input_mel.permute(0, 2, 1)

        embedding = self.embedding(input_ids)
        encoder_input = self.encoder_prenet(embedding)
        encoder_input = self.encoder_pe(encoder_input)

        decoder_input = self.decoder_prenet(input_mel)
        decoder_input = self.decoder_pe(decoder_input)

        device = input_ids.device

        if self.training:
            if attention_mask is None:
                attention_mask = self.get_padding_mask(input_ids).to(device)
            if decoder_attention_mask is None:
                decoder_attention_mask = self.get_padding_mask(input_mel[:, :, 0]).to(
                    device
                )
            d_lh_mask = self.get_decoder_mask(decoder_input.size(1)).to(device)
        else:
            d_lh_mask = None

        encoder_output, e_scores = self.encoder(encoder_input, attention_mask)
        decoder_output, d_scores, de_scores = self.decoder(
            decoder_input,
            encoder_output,
            decoder_attention_mask,
            d_lh_mask,
            attention_mask,
        )

        mel_output = self.mel_linear(decoder_output)
        mel_post_output = mel_output + self.postnet(mel_output)
        stop_output = self.stop_linear(decoder_output)

        return mel_output, mel_post_output, stop_output, e_scores, d_scores, de_scores
