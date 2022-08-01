import hydra
import torch

torch.autograd.set_detect_anomaly(True)

from datasets import get_dataloader, get_datasets
from models.pl_model import PLTransformerTTS
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from tokenizer import Tokenizer

torch.autograd.set_detect_anomaly(True)


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    seed_everything(cfg.TRAININGARGS.seed)

    tokenizer = Tokenizer(cfg.DATA.tokenizer_path)

    train_dataset, eval_dataset = get_datasets(tokenizer=tokenizer, **cfg.DATA.dataset)
    train_dataloader = get_dataloader(
        train_dataset, shuffle=True, **cfg.DATA.dataloader
    )
    eval_dataloader = get_dataloader(
        eval_dataset,
        shuffle=False,
        **cfg.DATA.dataloader,
    )

    if cfg.TRAININGARGS.load_from is not None:
        model = PLTransformerTTS.load_from_checkpoint(cfg.TRAININGARGS.load_from)
        print("load checkpoint at " + cfg.TRAININGARGS.load_from)
    else:
        model_config = dict(cfg.MODEL)
        model_config["vocab_size"] = tokenizer.vocab_size
        model = PLTransformerTTS(model_config, cfg.OPTIMIZER)

    wandb_logger = WandbLogger(**cfg.TRAININGARGS.wandb)

    callbacks = [ModelCheckpoint(**cfg.TRAININGARGS.ckpt)]

    trainer = Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        # plugins=DDPPlugin(find_unused_parameters=False),
        **cfg.TRAININGARGS.trainer,
    )

    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
