import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from aim import Image


def image_compare_reconstructions(originals, reconstructions):
    return torch.cat(
        [
            torch.cat([original, reconstruction], dim=1)
            for original, reconstruction in zip(originals, reconstructions)
        ],
        dim=2,
    )


class LitCifarTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model
        self.loss = torch.nn.MSELoss()

    def training_step(self, images, batch_idx):
        encoding = self.model.encode(images)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized, codebook_metrics = self.model.codebook.quantize(encoding)
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        reconstructions = self.model.decode(quantized)
        loss = self.loss(reconstructions, images) + codebook_metrics["loss_latent"]
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "perplexity", codebook_metrics["perplexity"], on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, images, batch_idx):
        reconstructions = self.model(images)
        reconstruction_loss = F.mse_loss(reconstructions, images)
        self.log("val_reconstruction_loss", reconstruction_loss)
        if batch_idx == 0:
            self.logger.experiment.track(
                Image(image_compare_reconstructions(images, reconstructions)),
                name="comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)


class LitGumbelCifarTrainer(LitCifarTrainer):
    def training_step(self, images, batch_idx):
        encoding = self.model.encode(images)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized, kl = self.model.codebook.quantize(encoding)
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        reconstructions = self.model.decode(quantized)
        loss = self.loss(reconstructions, images)
        loss = loss + kl
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
