from pathlib import Path
import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae.vqvae import OzeVQVAE

from src.trainer.oze import LitOzeTrainer
from oze.datamodule import OzeDataModule
from ..utils import parser, get_logger


class Experiment:
    exp_name = "oze-ozedata"
    dim_codebook = 32
    num_codebook = 256
    dataset_path = "datasets/oze/data_2020_2021.csv"
    lr = 1e-3
    T = 24 * 7

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = OzeDataModule(
            dataset_path=self.dataset_path,
            T=self.T,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        model = OzeVQVAE(
            num_codebook=self.num_codebook,
            dim_codebook=self.dim_codebook,
        )
        self.litmodule = LitOzeTrainer(model, lr=args.lr)

        # Load trainer
        self.logger = get_logger(self.exp_name)
        self.logger.experiment["hparams"] = vars(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor="val_reconstruction_loss",
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=self.logger,
            callbacks=[checkpoint_callback],
        )


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)

    # Train
    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
