from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae import CifarVQVAE
from vqvae.trainer import LITVqvae

from ..utils import parser, get_logger


EXP_NAME = "vqvae-cifar"
exp_name = f"{EXP_NAME}_{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}"

# Load model
dim_codebook = 32
num_codebook = 256
model = CifarVQVAE(
    num_codebook=num_codebook,
    dim_codebook=dim_codebook,
)
litmodule = LITVqvae(model, lr=3e-4)


def get_dataloader(args, split="train"):
    train = split == "train"

    def collate_fn(batch):
        return torch.stack([images for images, label in batch])

    dataset = datasets.CIFAR10(
        "./datasets/CIFAR10",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
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
