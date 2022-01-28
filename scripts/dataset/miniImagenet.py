from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae import VQVAE
from vqvae.trainer import LITVqvae

from src.dataset import MiniImagenet
from ..utils import parser, get_logger


EXP_NAME = "vqvae-miniImagenet"
exp_name = f"{EXP_NAME}_{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}"

# Load model
dim_codebook = 32
num_codebook = 256
channel_sizes = [16, 32, 32, dim_codebook]
strides = [2, 2, 1, 1]
model = VQVAE(
    num_codebook=num_codebook,
    dim_codebook=dim_codebook,
    in_channel=3,
    channel_sizes=channel_sizes,
    strides=strides,
)
litmodule = LITVqvae(model, lr=3e-4)


def get_dataloader(args, split="train"):
    dataset = torch.randn(1000, 3, 84, 84)
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    logger = get_logger(exp_name)
    # Define checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("checkpoints") / exp_name, monitor="train_loss"
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Load dataset
    dataloader_train = get_dataloader(args, split="train")
    dataloader_val = get_dataloader(args, split="val")

    trainer.fit(litmodule, dataloader_train, val_dataloaders=dataloader_val)
    # Save model weights
    torch.save(model.state_dict(), "model.pt")
