from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import OzeDataset, OzeDatasetRolling


class OzeDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_path: Path, T: int, batch_size: int, num_workers: int = 2
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._T = T
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.df = pd.read_csv(self._dataset_path)[5 * 24 :]
        OzeDataset.preprocess(self.df)
        self.dataset_train = OzeDatasetRolling(self.df, T=self._T, val=False)
        self.dataset_val = OzeDataset(self.df, T=self._T, val=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def export_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        u, y = list(zip(*batch))
        u = torch.stack(u).transpose(0, 1)
        y = torch.stack(y).transpose(0, 1)
        return u, y
