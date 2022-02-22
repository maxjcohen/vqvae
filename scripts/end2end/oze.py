from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vqvae.vqvae import OzeVQVAE

from src.oze.datamodule import OzeDataModule
from ..utils import parser, get_logger
from ..prior.oze import OzePrior
from src.trainer.oze import aim_fig_plot_ts


class Experiment:
    exp_name = "oze-end2end"
    dim_codebook = 16
    num_codebook = 8
    dim_command = 2
    T = 24 * 7
    dataset_path = "datasets/oze/data_2020_2021.csv"
    lr = 3e-3

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = OzeDataModule(
            dataset_path=self.dataset_path,
            T=self.T,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        vqvae = OzeVQVAE(
            num_codebook=self.num_codebook,
            dim_codebook=self.dim_codebook,
            codebook_flavor="gumbel",
        )
        prior = OzePrior(dim_command=self.dim_command, num_codebook=self.num_codebook)
        self.litmodule = LitOzeFull(vqvae=vqvae, prior=prior, lr=args.lr)

        # Load trainer
        self.logger = get_logger(self.exp_name)
        self.logger.experiment["hparams"] = vars(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor="train_prior",
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=self.logger,
            callbacks=[checkpoint_callback],
        )


def flatten_batches(array):
    return torch.cat(
        [sample for sample in torch.cat(array, dim=1).transpose(0, 1)], dim=0
    )


class LitOzeFull(pl.LightningModule):
    def __init__(self, vqvae, prior, lr=1e-3):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.lr = lr

    def training_step(self, batch, batch_idx):
        commands, observations = batch
        # Encode observation
        encoding = self.vqvae.encode(observations)
        # Sample quantized vector
        quantized, sample, codebook_metrics = self.vqvae.codebook.quantize(encoding)
        loss_posterior = codebook_metrics["loss_latent"]
        perplexity = codebook_metrics["perplexity"]
        indices = sample.argmax(-1)
        # Likelihood
        reconstructions = self.vqvae.decode(quantized)
        loss_reconstruction = F.mse_loss(reconstructions, observations)
        # Prior
        predictions = self.prior.forward(commands, indices)
        log_probs = torch.log(F.softmax(predictions, dim=-1))
        loss_prior = -torch.sum(sample * log_probs, dim=-1).mean()
        # TODO Replace with same torch builtin as in codebook
        # Addup losses
        loss = loss_reconstruction + loss_prior + loss_posterior
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_likelihood", loss_reconstruction, on_step=False, on_epoch=True)
        self.log("train_prior", loss_prior, on_step=False, on_epoch=True)
        self.log("train_posterior", loss_posterior, on_step=False, on_epoch=True)
        self.log("train_perplexity", perplexity, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        commands, observations = batch
        # Encoding
        encoding = self.vqvae.encode(observations)
        # Sample
        quantized, sample, _ = self.vqvae.codebook.quantize(encoding)
        indices = sample.argmax(-1)
        # Prior reconstruction
        prior_recons = self.prior.forward(commands, indices=indices)
        prior_recons = self.vqvae.decode(self.vqvae.codebook(prior_recons.argmax(-1)))
        # Prior sampling
        prior_sample = self.prior.forward(commands, indices=indices, generate=True)
        prior_sample = self.vqvae.decode(self.vqvae.codebook(prior_sample.argmax(-1)))
        # VQVAE reconstruction
        reconstructions = self.vqvae.decode(quantized)
        reconstruction_loss = F.mse_loss(reconstructions, observations)
        self.log("val_reconstruction_loss", reconstruction_loss)
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": observations[:, 1],
                        "predictions": reconstructions[:, 1],
                        "prior recons": prior_recons[:, 1],
                        "prior sample": prior_sample[:, 1],
                    }
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return observations, reconstructions, prior_recons, prior_sample

    def validation_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            observations, reconstructions, prior_recons, prior_sample = map(
                flatten_batches, zip(*outputs)
            )
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {
                        "observations": observations,
                        "predictions": reconstructions,
                        "prior recons": prior_recons,
                        "prior sample": prior_sample,
                    }
                ),
                name="full-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)

    # Train
    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
