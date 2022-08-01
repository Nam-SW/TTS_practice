from typing import Dict

import torch
import torch.nn as nn
import wandb
from models.transformer_tts import TransformerTTS
from pytorch_lightning import LightningModule


def get_transformer_warmup_scheduler(
    optimizer,
    hidden_size,
    num_warmup_steps=4000,
):
    hidden_size = torch.tensor(hidden_size)

    def lr_lambda(step):
        arg1 = torch.rsqrt(torch.tensor(step))
        arg2 = torch.tensor(step * (num_warmup_steps**-1.5))

        return torch.rsqrt(hidden_size) * torch.minimum(arg1, arg2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PLTransformerTTS(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        class_weight: float = 8.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerTTS(**model_config)
        self.optimizer_config = optimizer_config

        self.mel_loss = nn.L1Loss()
        self.stop_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight))

    def forward(self, inputs):
        return self.model(**inputs)

    def loss_function(
        self,
        mel_output,
        mel_post_output,
        stop_output,
        mel_label,
        stop_label,
    ):
        if mel_output.size(1) != mel_label.size(1):
            mel_label = mel_label.permute(0, 2, 1)

        mel_loss = (
            self.mel_loss(mel_output, mel_label).mean()
            + self.mel_loss(mel_post_output, mel_label).mean()
        )
        stop_loss = self.stop_loss(
            stop_output.squeeze(-1),
            stop_label,
        ).mean()
        return mel_loss, stop_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        mel_output, mel_post_output, stop_output, _, _, _ = self(x)
        mel_loss, stop_loss = self.loss_function(
            mel_output, mel_post_output, stop_output, **y
        )
        global_loss = mel_loss + stop_loss

        lr_scheduler = self.lr_schedulers()

        log_dict = {
            "train/e_alpha": self.model.encoder_pe.alpha,
            "train/d_alpha": self.model.decoder_pe.alpha,
            "train/mel_loss": mel_loss,
            "train/stop_loss": stop_loss,
            "train/global_loss": global_loss,
            "train/lr": lr_scheduler.get_last_lr()[0],
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return global_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        mel_output, mel_post_output, stop_output, e_score, d_score, de_score = self(x)
        mel_loss, stop_loss = self.loss_function(
            mel_output, mel_post_output, stop_output, **y
        )
        global_loss = mel_loss + stop_loss

        log_dict = {
            "eval/mel_loss": mel_loss,
            "eval/stop_loss": stop_loss,
            "eval/global_loss": global_loss,
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return global_loss, e_score, d_score, de_score

    def validation_epoch_end(self, validation_step_outputs):
        _, e_score, d_score, de_score = validation_step_outputs[0]

        for score, name in zip([e_score, d_score, de_score], ["enc", "dec", "enc-dec"]):
            tags = []
            attn_plots = []
            for i in range(self.model.num_layers):
                tag = f"layer{i + 1}"
                img = wandb.Image(score[i, 0].cpu(), caption=tag)
                tags.append(tag)
                attn_plots.append(img)

            self.logger.log_image(
                key=name + "_attn_plots",
                images=attn_plots,
                caption=tags,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            betas=self.optimizer_config["betas"],
            eps=self.optimizer_config["esp"],
        )
        scheduler = {
            "scheduler": get_transformer_warmup_scheduler(
                optimizer,
                self.model.hidden_size,
                self.optimizer_config["warmup_steps"],
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
