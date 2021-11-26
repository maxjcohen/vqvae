import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

from vqvae import VQVAE

import src.metrics as metrics
from src.utils import export_distances

# Training settings
EPOCHS = 1
BATCH_SIZE = 16
NUM_WORKERS = 4

# Dataset settings
C = 3
W = 28
H = 28

# Model settings
N_CODEBOOK = 256
DIM_CODEBOOK = 32
channel_sizes = [16, DIM_CODEBOOK]
strides = [2, 2]


def get_dataloader(train=True):
    dataset = datasets.CIFAR10(
        "./datasets/CIFAR10",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS,
    )


def train(model, dataloader):
    # Traininp loop
    class LITVqvae(pl.LightningModule):
        def __init__(self, model, lr=1e-3):
            super().__init__()
            self.lr = lr
            self.model = model
            self.hist = {"loss": [], "rank": []}

        def training_step(self, batch, batch_idx):
            images, labels = batch
            encodings = self.model.encode(images)
            qt = self.model.quanticize(encodings)
            loss_latent = torch.nn.functional.mse_loss(encodings, qt)
            n_uniques = len(qt.permute(0, 2, 3, 1).view(-1, DIM_CODEBOOK).unique(dim=0))
            qt = encodings + (qt - encodings).detach()
            reconstructions = self.model.decode(qt)
            loss = torch.nn.functional.mse_loss(images, reconstructions)
            loss = loss + loss_latent
            self.hist["loss"].append(loss.item())
            self.hist["rank"].append(n_uniques)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(self.lr))

    # Training
    train_model = LITVqvae(model, lr=1e-3)
    trainer = pl.Trainer(max_epochs=EPOCHS, gpus=1)
    trainer.fit(train_model, dataloader)

    # Save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vqvae helper script.")
    parser.add_argument("actions", nargs="+")
    args = parser.parse_args()
    # Load model
    model = VQVAE(
        in_channel=3,
        channel_sizes=channel_sizes,
        n_codebook=N_CODEBOOK,
        dim_codebook=DIM_CODEBOOK,
        strides=strides,
    )
    dataloader_train = get_dataloader(train=True)

    if "train" in args.actions:
        train(model, dataloader_train)
        dataloader_val = get_dataloader(train=False)
        reconstruction_cost = metrics.reconstruction(model, dataloader_val)
        print(f"Reconstruction cost: {reconstruction_cost.item()}.")

    if "export" in args.actions:
        try:
            model.load_state_dict(torch.load("model.pt"))
        except FileNotFoundError:
            print("Model weights not found, starting training.")
            train(model, dataloader)
        distances = export_distances(model, dataloader_train, reduction="none")
        print(f"Export distances with shape {distances.shape}.")
        torch.save(distances, "distances.pt")
