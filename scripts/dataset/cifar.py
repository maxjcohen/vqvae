from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae import CifarVQVAE

from src.trainer.cifar import LitCifarTrainer
from ..utils import parser, get_logger


class CifarDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: Path, batch_size: int, num_workers: int = 2):
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage=None):
        self.dataset_train = datasets.CIFAR10(
            self._dataset_path,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.dataset_val = datasets.CIFAR10(
            self._dataset_path,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        return torch.stack([images for images, label in batch])


class Experiment:
    exp_name = "vqvae-cifar-gumbel"
    dim_codebook = 32
    num_codebook = 256
    dataset_path = "./datasets/CIFAR10"
    lr = 3e-4

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = CifarDataModule(
            dataset_path=self.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        model = CifarVQVAE(
            num_codebook=self.num_codebook,
            dim_codebook=self.dim_codebook,
            codebook_flavor="gumbel",
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
