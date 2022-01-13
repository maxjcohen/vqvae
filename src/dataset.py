import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class MiniImagenet(Dataset):
    """Dataset for mini-Imagenet.

    This dataset was introducedd in https://arxiv.org/abs/1606.04080, the actual files
    were downloaded from https://github.com/tristandeleu/pytorch-meta. This Dataset
    loads the selected split's labels and images. The `__getitem__` method treats the
    dataset as a stack of all classes. For this reason, it should be used with a
    dataloader set to suffle samples during training.
    """

    def __init__(self, root: Path, split: str = "train"):
        """Load images and labels.

        Parameters
        ----------
        root: Path to the dataset folder.
        split: One of `"train"`, `"test"` or `"val"`. Default is `"train"`.
        """
        # Select split
        _allowed_splits = ["train", "test", "val"]
        assert (
            split in _allowed_splits
        ), f"Invalid split {split}, must be in {_allowed_splits}."
        # Load labels
        root = Path(root)
        with open(root / f"{split}_labels.json", "r") as json_file:
            self.labels = json.load(json_file)
        # Load images
        f = h5py.File(root / f"{split}_data.hdf5", "r")
        self.dset_images = f["datasets"]

    def __len__(self):
        return self.n_classes * self.n_image_per_class

    def __getitem__(self, item):
        # Select image
        image = self.dset_images[self.labels[item // self.n_image_per_class]][
            item % self.n_image_per_class
        ]
        # Normalize
        image = image / 255
        # Swap axis to channel first
        image = image.transpose(2, 0, 1)
        # Convert to Tensor
        return torch.Tensor(image)

    @property
    def n_classes(self):
        return len(self.labels)

    @property
    def n_image_per_class(self):
        return self.dset_images[self.labels[0]].shape[0]
