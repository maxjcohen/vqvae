"""Export module.

This module define functions for exporting various latent vectors of the vqvae model.
All export function defined below follow the same signature:

Parameters
----------
model: Instance of the vqvae model.
dataloader: Iterable over the dataset.
export_path: Path to save the exported tensor.
device_model: Torch device used to compute inference. Default is `"cpu"`.
device_storage: Torch device used to store the exported tensor. Default is `"cpu"`.
"""
import torch
from tqdm import tqdm


def export_iteration(export_function):
    @torch.no_grad()
    def modified_function(
        model, dataloader, export_path, device_model="cpu", device_storage="cpu"
    ):
        model.to(device_model)
        export_values = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            export_values.append(
                export_function(model, batch, device=device_model).to(device_storage)
            )
        export_values = torch.cat(export_values, dim=0)
        print(f"Exporting tensor with shape {export_values.shape} to {export_path}.")
        torch.save(export_values, export_path)

    return modified_function


@export_iteration
def distances(model, batch, device="cpu"):
    """Export distances to each codebook."""
    images, _ = batch
    encodings = model.encode(images.to(device))
    distances = model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
    return distances


@export_iteration
def encodings(model, batch, device="cpu"):
    """Export encodings (i.e. Z_e)."""
    images, _ = batch
    encodings = model.encode(images.to(device))
    return encodings


@export_iteration
def indexes(model, batch, device="cpu"):
    """Export indexes of the closest codebook."""
    images, _ = batch
    encodings = model.encode(images.to(device))
    distances = model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
    indexes = distances.argmin(dim=-1)
    return indexes


@export_iteration
def codebooks(model, batch, device="cpu"):
    """Export codebook associated with each encoding."""
    images, _ = batch
    encodings = model.encode(images.to(device))
    codebooks = model.quantize(encodings)
    return codebooks
