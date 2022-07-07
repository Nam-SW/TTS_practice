from typing import Dict

import torch
import torch.nn.functional as F
from models.discriminator import Discriminator
from models.generator import Generator
from pytorch_lightning import LightningModule


class MelGAN(LightningModule):
    def __init__(
        self,
        generator_config: Dict,
        discriminator_config: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)

    def forward(self, s, x):
        gen_output = self.generator(s)
        fake_disc_output = self.discriminator(gen_output)
        real_disc_output = self.discriminator(x)
        return gen_output, real_disc_output, fake_disc_output

    def gen_loss(self, real_disc, fake_disc):
        gen_loss = 0
        for disc_output in fake_disc[0]:
            gen_loss += -disc_output.mean()

        for feature_r, feature_f in zip(real_disc[1], fake_disc[1]):
            for r, f in zip(feature_r, feature_f):
                gen_loss += 10 * F.l1_loss(r, f)

        return gen_loss

    def disc_loss(self, real_disc, fake_disc):
        disc_loss = 0
        for output_r, output_f in zip(real_disc[0], fake_disc[0]):
            disc_loss += F.relu(1 - output_r.mean()) + F.relu(1 + output_f.mean())

        return disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        s, x = batch

        gen_output, real_disc_output, fake_disc_output = self(s, x)

        # train generator
        if optimizer_idx == 0:
            g_loss = self.gen_loss(real_disc_output, fake_disc_output)

            self.log(
                "train/gen_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.disc_loss(real_disc_output, fake_disc_output)

            self.log(
                "train/disc_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return d_loss

    def validation_step(self, batch, batch_idx):
        s, x = batch

        gen_output, real_disc_output, fake_disc_output = self(s, x)
        g_loss = self.gen_loss(real_disc_output, fake_disc_output)

        gen_output, real_disc_output, fake_disc_output = self(s, x)
        d_loss = self.disc_loss(real_disc_output, fake_disc_output)

        self.log(
            "eval/gen_loss",
            g_loss,
            on_epoch=True,
            logger=True,
        )

        self.log(
            "eval/disc_loss",
            d_loss,
            on_epoch=True,
            logger=True,
        )

        return g_loss + d_loss

    def configure_optimizers(self):
        lr = 1e-4
        betas = [0.5, 0.9]

        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=betas,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=betas,
        )
        return [opt_g, opt_d], []
