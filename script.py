import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from vqvae import VQVAE

import src.metrics as metrics
from src.utils import export_distances

# Dataset settings
C = 3
W = 28
H = 28

# Model settings
N_CODEBOOK = 256
DIM_CODEBOOK = 32
channel_sizes = [16, 32, DIM_CODEBOOK]
strides = [2, 2, 1]


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


def train(args, model, dataloader):
    # Traininp loop
    class LITVqvae(pl.LightningModule):
        def __init__(self, model, lr=1e-3):
            super().__init__()
            self.lr = lr
            self.model = model
            self.hist = {"loss": [], "rank": []}
            self.loss = torch.nn.BCEWithLogitsLoss()

        def training_step(self, batch, batch_idx):
            images, _ = batch
            encodings = self.model.encode(images)
            qt = self.model.quantize(encodings)
            loss_latent = torch.nn.functional.mse_loss(encodings, qt)
            n_uniques = len(qt.permute(0, 2, 3, 1).view(-1, DIM_CODEBOOK).unique(dim=0))
            qt = encodings + (qt - encodings).detach()
            reconstructions = self.model.decode(qt)
            loss = self.loss(reconstructions, images)
            loss = loss + loss_latent
            self.hist["loss"].append(loss.item())
            self.hist["rank"].append(n_uniques)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(self.lr))

    # Training
    train_model = LITVqvae(model, lr=1e-3)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1)
    trainer.fit(train_model, dataloader)

    for idx, (loss_name, loss_values) in enumerate(train_model.hist.items()):
        plt.subplot(len(train_model.hist), 1, idx + 1)
        loss_values = np.array(loss_values)
        plt.plot(loss_values, alpha=0.4)
        plt.plot(
            np.arange(1, len(loss_values), len(dataloader)),
            loss_values.reshape(args.epochs, len(dataloader)).mean(-1),
        )
        plt.xlabel("Epochs")
        plt.ylabel(loss_name)
    plt.savefig("hist.jpg")

    # Save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
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
    args = parser.parse_args()
    device = torch.device('cuda:0')
    # Load model
    model = VQVAE(
        in_channel=3,
        channel_sizes=channel_sizes,
        n_codebook=N_CODEBOOK,
        dim_codebook=DIM_CODEBOOK,
        strides=strides,
    )
    dataloader_train = get_dataloader(args, train=True)

    if "train" in args.actions:
        train(args, model, dataloader_train)
        dataloader_val = get_dataloader(args, train=False)
        model.to(device)
        reconstruction_cost = metrics.reconstruction(model, dataloader_val, device)
        print(f"Reconstruction cost: {reconstruction_cost.item()}.")

    if "export" in args.actions:
        try:
            model.load_state_dict(torch.load("model.pt"))
        except FileNotFoundError:
            print("Model weights not found, starting training.")
            train(args, model, dataloader)
        distances = export_distances(model, dataloader_train, reduction="none")
        print(f"Export distances with shape {distances.shape}.")
        torch.save(distances, "distances.pt")
