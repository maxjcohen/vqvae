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
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized = self.model.codebook.quantize(encoding)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        reconstructions = self.model.decode(quantized)
        loss = self.loss(reconstructions, images)
        loss = loss + loss_latent
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, images, batch_idx):
        reconstructions = self.model(images)
        reconstruction_loss = F.mse_loss(reconstructions, images)
        self.log("val_reconstruction_loss", reconstruction_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)
