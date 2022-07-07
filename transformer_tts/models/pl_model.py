from turtle import st
from typing import Dict, Union

import torch
import torch.nn.functional as F
from models.transformer_tts import TransformerTTS
from pytorch_lightning import LightningModule


def get_transformer_warmup_scheduler(
    optimizer,
    hidden_size,
    num_warmup_steps=4000,
):
    def lr_lambda(step):
        arg1 = torch.rsqrt(step)
        arg2 = step * (num_warmup_steps**-1.5)

        return torch.rsqrt(hidden_size) * torch.minimum(arg1, arg2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PLTransformerTTS(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        class_weight: Union[int, float] = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerTTS(**model_config)
        self.optimizer_config = optimizer_config
        self.class_weight = torch.tensor(class_weight)

    def forward(self, input_ids, spectrogram):
        return self.model(input_ids, spectrogram)

    def loss_function(self, mel_output, stop_output, mel_label, stop_label):
        mel_loss = F.mse_loss(mel_output, mel_label)
        stop_loss = F.binary_cross_entropy(
            stop_output, stop_label, weight=self.class_weight
        )
        return mel_loss, stop_loss

    def training_step(self, batch, batch_idx):
        input_ids, spectrogram = batch

        mel_output, stop_output = self(input_ids, spectrogram)
        mel_loss, stop_loss = self.loss_function(mel_output, stop_output)

        log_dict = {
            "train/mel_loss": mel_loss,
            "train/stop_loss": stop_loss,
            "train/global_loss": mel_loss + stop_loss,
        }

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def validation_step(self, batch):
        input_ids, spectrogram = batch

        mel_output, stop_output = self(input_ids, spectrogram)
        mel_loss, stop_loss = self.loss_function(mel_output, stop_output)

        log_dict = {
            "eval/mel_loss": mel_loss,
            "eval/stop_loss": stop_loss,
            "eval/global_loss": mel_loss + stop_loss,
        }

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            betas=self.optimizer_config["betas"],
        )
        lr_scheduler = get_transformer_warmup_scheduler(
            optimizer,
            self.model.hidden_size,
            self.optimizer_config["warmup_steps"],
        )
        return optimizer, lr_scheduler
