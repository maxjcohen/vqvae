from pathlib import Path
import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae.vqvae import OzeVQVAE

from src.trainer.oze import LitOzeTrainer
from oze.datamodule import OzeDataModule
from ..utils import parser, get_logger


exp_name = "oze-ozedata"

# Load model
dim_codebook = 32
num_codebook = 256
model = OzeVQVAE(
    num_codebook=num_codebook,
    dim_codebook=dim_codebook,
)


def get_datamodule(args):
    return OzeDataModule(
        dataset_path="datasets/oze/data_2020_2021.csv",
        T=24*7,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    args = parser.parse_args()
    args.lr = args.lr or 1e-3
    litmodule = LitOzeTrainer(model, lr=args.lr)

    # Load logger
    logger = get_logger(exp_name)
    logger.experiment["hparams"] = vars(args)

    # Define checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("checkpoints") / exp_name,
        filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
        monitor="val_reconstruction_loss",
        save_last=True,
    )

    # Load trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Load dataset
    datamodule = get_datamodule(args)

    # Train
    trainer.fit(litmodule, datamodule=datamodule)
