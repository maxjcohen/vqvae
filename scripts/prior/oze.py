from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

from vqvae.vqvae import OzeVQVAE

from src.oze.dataset import OzeDataset
from ..utils import parser, get_logger


class OzePrior(nn.Module):
    latent_dim = 32

    def __init__(self, dim_command, num_codebook, tau=0.1):
        super().__init__()
        p_dropout = 0.2
        self.kernel = nn.GRUCell(input_size=self.latent_dim, hidden_size=num_codebook)
        self.input_embedding = nn.GRU(
            input_size=dim_command,
            hidden_size=self.latent_dim,
            num_layers=3,
            dropout=p_dropout,
        )
        self.num_codebook = num_codebook
        self.tau = tau
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, commands, indices, generate=False, step_sample=False):
        commands = self.input_embedding(commands)[0]
        indices_onehot = F.one_hot(indices, num_classes=self.num_codebook).float()
        indices = self.dropout(indices_onehot)
        if generate:
            h_t = indices_onehot[0]
            outputs = [h_t]
            for command in commands[1:]:
                h_t = self.kernel(command, h_t) / self.tau
                h_t = (
                    F.softmax(h_t, dim=-1).multinomial(1).squeeze()
                    if step_sample
                    else h_t.argmax(-1)
                )
                h_t = F.one_hot(h_t, num_classes=self.num_codebook).float()
                outputs.append(h_t)
            return torch.stack(outputs, dim=0)
        outputs = torch.empty(indices_onehot.shape, device=indices_onehot.device)
        outputs[0] = indices_onehot[0]
        commands = commands[1:].view(-1, self.latent_dim)
        indices_onehot = indices_onehot[:-1].view(-1, self.num_codebook)
        outputs[1:] = (
            self.kernel(commands, indices_onehot).view(167, -1, self.num_codebook)
            / self.tau
        )
        return outputs


class LitOzePrior(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        commands, _, indices = batch
        predictions = self.model(commands, indices)
        loss = F.cross_entropy(predictions.permute(1, 2, 0), indices.permute(1, 0))
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)


class OzeIndicesDataset(OzeDataset):
    def __init__(self, df, indices, val=False):
        super().__init__(df, T=indices.shape[0], val=val)
        self.indices = indices

    def __getitem__(self, idx):
        return (
            self.u[idx * self.T : (idx + 1) * self.T],
            self.y[idx * self.T : (idx + 1) * self.T],
            self.indices[:, idx],
        )


class OzeIndicesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        indices_path: Path,
        batch_size: int,
        num_workers: int = 2,
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._indices_path = indices_path
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage=None):
        self.df = pd.read_csv(self._dataset_path)[5 * 24 :]
        OzeIndicesDataset.preprocess(self.df)
        indices = torch.load(self._indices_path)
        self.dataset_train = OzeIndicesDataset(self.df, indices, val=False)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        commands, observations, indices = list(zip(*batch))
        commands = torch.stack(commands).transpose(0, 1)
        observations = torch.stack(observations).transpose(0, 1)
        indices = torch.stack(indices).transpose(0, 1)
        return commands, observations, indices


class Experiment:
    exp_name = "oze-prior"
    dim_codebook = 16
    num_codebook = 8
    dim_command = 2
    dataset_path = "datasets/oze/data_2020_2021.csv"
    indices_path = "indexes_oze.pt"
    lr = 3e-3

    def __init__(self, args):
        args.lr = args.lr or self.lr

        # Load dataset
        self.datamodule = OzeIndicesDataModule(
            dataset_path=self.dataset_path,
            indices_path=self.indices_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        model = OzePrior(dim_command=self.dim_command, num_codebook=self.num_codebook)
        self.litmodule = LitOzePrior(model, lr=args.lr)

        # Load trainer
        self.logger = get_logger(self.exp_name)
        self.logger.experiment["hparams"] = vars(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor="train_loss",
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=self.logger,
            callbacks=[checkpoint_callback],
        )


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)

    # Train
    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
