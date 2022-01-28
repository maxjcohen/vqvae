import torch
import pytorch_lightning as pl


class LITVqvae(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model
        self.loss = torch.nn.BCEWithLogitsLoss()

    def training_step(self, images, batch_idx):
        encodings = self.model.encode(images)
        qt = self.model.quantize(encodings)
        loss_latent = torch.nn.functional.mse_loss(encodings, qt)
        n_uniques = len(
            qt.permute(0, 2, 3, 1).view(-1, self.model.dim_codebook).unique(dim=0)
        )
        qt = encodings + (qt - encodings).detach()
        reconstructions = self.model.decode(qt)
        loss = self.loss(reconstructions, images)
        loss = loss + loss_latent
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rank", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, images, batch_idx):
        # Forward
        encodings = self.model.encode(images)
        qt = self.model.quantize(encodings)
        reconstructions = torch.sigmoid(self.model.decode(qt))
        # Compute loss
        n_uniques = len(
            qt.permute(0, 2, 3, 1).view(-1, self.model.dim_codebook).unique(dim=0)
        )
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructions, images)
        # Log
        self.log("val_reconstruction_loss", reconstruction_loss)
        self.log("val_rank", n_uniques)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(self.lr))
