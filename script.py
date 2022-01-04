import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from vqvae import VQVAE
from vqvae.trainer import LITVqvae

import src.export as export


try:
    from aim.pytorch_lightning import AimLogger

    logger = AimLogger(experiment="vqvae", system_tracking_interval=None)
except ImportError:
    logger = None
print(f"Using logger {logger}.")


def configure_parser():
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
    parser.add_argument(
        "--num-codebook",
        default=256,
        type=int,
        help="Number of codebooks.",
    )
    parser.add_argument(
        "--dim-codebook",
        default=32,
        type=int,
        help="Dimension of codebook vectors.",
    )
    return parser


def get_dataloader(args, train=True):
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
    )


if __name__ == "__main__":
    args = configure_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    channel_sizes = [16, 32, 32, args.dim_codebook]
    strides = [2, 2, 1, 1]
    model = VQVAE(
        in_channel=3,
        channel_sizes=channel_sizes,
        n_codebook=args.num_codebook,
        dim_codebook=args.dim_codebook,
        strides=strides,
    )
    train_model = LITVqvae(model, lr=3e-4)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, logger=logger)

    # Load dataset
    dataloader_train = get_dataloader(args, train=True)
    dataloader_val = get_dataloader(args, train=False)

    if "train" in args.actions:
        trainer.fit(train_model, dataloader_train, val_dataloaders=dataloader_val)
        # Save model
        torch.save(model.state_dict(), "model.pt")

    if "export" in args.actions:
        model.load_state_dict(torch.load("model.pt"))
        export.indexes(model, dataloader_train, export_path="indexes.pt", device_model=device)
        export.distances(model, dataloader_train, export_path="distances.pt", device_model=device)
        export.codebooks(model, dataloader_train, export_path="codebooks.pt", device_model=device)

    if "eval" in args.actions:
        model.load_state_dict(torch.load("model.pt"))
        trainer.validate(train_model, dataloaders=dataloader_val)
