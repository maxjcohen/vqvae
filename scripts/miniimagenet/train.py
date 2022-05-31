from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger

from src.trainer.cifar import LitCifarTrainer
from src.dataset import MiniImagenetDataModule
from ..utils import parser

parser.add_argument("--flavor", default="classic", type=str, help="Codebook flavor.")


class Experiment:
    exp_name = "vqvae-miniimagenet-train"
    dim_codebook = 32
    num_codebook = 64
    dataset_path = "./datasets/miniImagenet"
    lr = 1e-3

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = MiniImagenetDataModule(
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
