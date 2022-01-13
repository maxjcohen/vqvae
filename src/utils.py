from pathlib import Path

from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm


def convert_hdf5(dataset_path: Path, export_path: Path, dataset_name: str = None):
    dataset_path = Path(dataset_path)
    dataset_name = dataset_name or dataset_name.name
    # Load images
    images = []
    for image_path in tqdm(dataset_path.glob("*.jpg"), total=60000):
        images.append(np.asarray(Image.open(image_path)))
        import pdb

        pdb.set_trace()
    images = np.array(images)
    # Export dataset to hdf5
    with h5py.File(export_path, "w") as f:
        dset = f.create_dataset(dataset_name, data=images)
    print(f"Exported dataset with shape {images.shape}.")


def load_hsf5(dataset_path: Path, dataset_name: str = None) -> np.ndarray:
    with h5py.File(dataset_path, "r") as f:
        dataset_name = dataset_name or list(f.keys())[0]
        return np.array(f[dataset_name])


if __name__ == "__main__":
    dataset_path = "datasets/miniImagenet/images/"
    export_path = "datasets/miniImagenet/miniImagenet.hdf5"
    convert_hdf5(
        dataset_path=dataset_path, export_path=export_path, dataset_name="miniImagenet"
    )
