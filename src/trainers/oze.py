import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import plotly.graph_objects as go
from aim import Figure


class OzeTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model

    def training_step(self, batch, batch_idx):
        u, y = batch
        encoding = self.model.encode(u)
        quantized = self.model.codebook.quantize(encoding)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        reconstructions = self.model.decode(quantized)
        loss = F.mse_loss(reconstructions, y) + loss_latent
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        reconstructions = self.model(u)
        reconstruction_loss = F.mse_loss(reconstructions, y)
        self.log("val_reconstruction_loss", reconstruction_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)
