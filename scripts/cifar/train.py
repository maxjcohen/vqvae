from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger

from src.trainer.cifar import LitCifarTrainer
from ..utils import parser

parser.add_argument("--flavor", default="classic", type=str, help="Codebook flavor.")


class CifarDataModule(pl.LightningDataModule):
    mean: float = 0.4734
    std: float = 0.2516

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int,
        num_workers: int = 2,
        standardize: bool = False,
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._transforms = transforms.ToTensor()
        if standardize:
            self._transforms = transforms.Compose(
                [self._transforms, transforms.Normalize(mean=self.mean, std=self.std)]
            )

    def setup(self, stage=None):
        self.dataset_train = datasets.CIFAR10(
            self._dataset_path,
            train=True,
            download=True,
            transform=self._transforms,
        )
        self.dataset_val = datasets.CIFAR10(
            self._dataset_path,
            train=False,
            download=True,
            transform=self._transforms,
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

    def rescale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


class Experiment:
    exp_name = "vqvae-cifar-train"
    dim_codebook = 32
    num_codebook = 128
    dataset_path = "./datasets/CIFAR10"
    lr = 1e-3

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = CifarDataModule(
            dataset_path=self.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        if args.load_path:
            self.litmodule = LitCifarTrainer.load_from_checkpoint(args.load_path)
        else:
            self.litmodule = LitCifarTrainer(
                num_codebook=self.num_codebook,
                dim_codebook=self.dim_codebook,
                codebook_flavor=args.flavor,
                lr=args.lr,
            )

        # Load trainer
        self.logger = AimLogger(
            experiment=self.exp_name,
            system_tracking_interval=None,
            log_system_params=False,
        )
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
