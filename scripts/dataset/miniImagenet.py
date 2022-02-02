from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae import CifarVQVAE

from src.trainer.cifar import LitCifarTrainer
from src.dataset import MiniImagenet
from ..utils import parser, get_logger


class ImagenetDataModule(pl.LightningDataModule):
    _dataset_path = "datasets/MiniImagenet/miniimagenet/"

    def __init__(self, batch_size: int, num_workers: int = 2):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage=None):
        self.dataset_train = MiniImagenet(root=self._dataset_path, split="train")
        self.dataset_val = MiniImagenet(root=self._dataset_path, split="val")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )


class Experiment:
    exp_name = "vqvae-miniImagenet"
    dim_codebook = 32
    num_codebook = 128
    lr = 1e-4

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = ImagenetDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        model = CifarVQVAE(
            num_codebook=self.num_codebook,
            dim_codebook=self.dim_codebook,
        )
        self.litmodule = LitCifarTrainer(model, lr=args.lr)

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
