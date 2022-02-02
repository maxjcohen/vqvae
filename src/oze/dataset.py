import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import compute_occupancy


class OzeDataset(Dataset):
    input_columns = [
        "temperature_exterieure",
        "RHUM",
    ]
    target_columns = [
        "humidite",
    ]

    def __init__(self, df, T, val=False):
        self.T = T
        self._normalization_const = {}

        self.u = df[self.input_columns][df["val"] == val].values
        self.u = torch.Tensor(self.normalize(self.u, label="command"))
        self.y = df[self.target_columns][df["val"] == val].values
        self.y = torch.Tensor(self.normalize(self.y, label="observation"))

    def __getitem__(self, idx):
        return (
            self.u[idx * self.T : (idx + 1) * self.T],
            self.y[idx * self.T : (idx + 1) * self.T],
        )

    def __len__(self):
        return len(self.y) // self.T

    @staticmethod
    def preprocess(df):
        # Convert timestamp to datetime
        df["datetime"] = df["date"].apply(
            lambda date: datetime.datetime.fromtimestamp(date)
        )
        # Add occupancy
        df["occupancy"] = df["datetime"].apply(compute_occupancy)
        # Flag validation for 2021 data, starts on first monday
        df["val"] = df["datetime"] > datetime.datetime(year=2021, month=1, day=4)

    def normalize(self, array, label=None):
        array_mean = array.mean(axis=0, keepdims=True)
        array_std = array.std(axis=0, keepdims=True)
        if label:
            self._normalization_const[label] = (array_mean, array_std)
        return (array - array_mean) / array_std

    def rescale(self, array, label):
        try:
            array_mean, array_std = self._normalization_const[label]
        except KeyError:
            raise NameError(f"Can't rescale array with unknown label {label}.")
        return array * array_std + array_mean
