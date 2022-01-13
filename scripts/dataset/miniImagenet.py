import argparse
import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint

from vqvae import VQVAE
from vqvae.trainer import LITVqvae

import src.export as export
from src.dataset import MiniImagenet


EXP_NAME = "vqvae-miniImagenet"
exp_name = f"{EXP_NAME}_{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}"

parser = argparse.ArgumentParser(description="vqvae helper script.")
parser.add_argument("actions", nargs="+")
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs for training. Default is 100.",
)
parser.add_argument(
    "--num-workers",
    default=1,
    type=int,
    help="Number of workers for dataloaders. Default is 1.",
)
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size for dataloaders. Default is 16.",
)
parser.add_argument("--device", type=str, help="Specify torch device.")

try:
    from aim.pytorch_lightning import AimLogger

    logger = AimLogger(experiment=exp_name, system_tracking_interval=None)
except ImportError:
    logger = None
print(f"Using logger {logger}.")

# Load model
dim_codebook = 32
num_codebook = 256
channel_sizes = [16, 32, 32, dim_codebook]
strides = [2, 2, 1, 1]
model = VQVAE(
    in_channel=3,
    channel_sizes=channel_sizes,
    n_codebook=num_codebook,
    dim_codebook=dim_codebook,
    strides=strides,
)
train_model = LITVqvae(model, lr=3e-4)

# Define checkpoints
checkpoint_callback = ModelCheckpoint(
    "checkpoints/", monitor="train_loss", filename=exp_name
)


def get_dataloader(args, split="train"):
    dataset = MiniImagenet(
        root="datasets/MiniImagenet/miniimagenet/",
        split=split,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    # Define torch device
    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, RichModelSummary()],
    )

    # Load dataset
    dataloader_train = get_dataloader(args, split="train")
    dataloader_val = get_dataloader(args, split="val")

    if "train" in args.actions:
        trainer.fit(train_model, dataloader_train, val_dataloaders=dataloader_val)
        # Save model weights
        torch.save(model.state_dict(), "model.pt")

    if "export" in args.actions:
        model.load_state_dict(torch.load("model.pt"))
        export.encodings(
            model, dataloader_train, export_path="encodings.pt", device_model=device
        )

    if "eval" in args.actions:
        model.load_state_dict(torch.load("model.pt"))
        trainer.validate(train_model, dataloaders=dataloader_val)
