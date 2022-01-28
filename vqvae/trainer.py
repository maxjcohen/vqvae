import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LITVqvae(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model
        self.loss = torch.nn.MSELoss()

    def training_step(self, images, batch_idx):
        encoding = self.model.encode(images)
        quantized = self.model.quantize(encoding)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        reconstructions = self.model.decode(quantized)
        loss = self.loss(reconstructions, images)
        loss = loss + loss_latent
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, images, batch_idx):
        # Forward
        encoding = self.model.encode(images)
        quantized = self.model.quantize(encoding)
        reconstructions = torch.sigmoid(self.model.decode(quantized))
        # Compute loss
        reconstruction_loss = F.mse_loss(reconstructions, images)
        # Log
        self.log("val_reconstruction_loss", reconstruction_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(self.lr))
