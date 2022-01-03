from pathlib import Path

import torch
from tqdm import tqdm


@torch.no_grad()
def export_distances_batch(model, dataloader, export_path="."):
    export_path = Path(export_path)
    export_path.mkdir(exist_ok=True)
    for idx_batch, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        encodings = model.encode(images)
        distances = model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
        torch.save(distances, export_path / f"batch_{idx_batch}.pt")


@torch.no_grad()
def export_distances(model, dataloader, export_path):
    export_path = Path(export_path)
    distances = []
    for idx_batch, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        encodings = model.encode(images)
        distances.append(model.emb.compute_distances(encodings.permute(0, 2, 3, 1)))
    distances = torch.cat(distances, dim=0)
    print(f"Exporting distances with shape {distances.shape}.")
    torch.save(distances, export_path)


@torch.no_grad()
def export_indexes(model, dataloader, export_path, device="cuda:0"):
    model.to(device)
    export_path = Path(export_path)
    indexes = []
    for idx_batch, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        encodings = model.encode(images.to(device))
        distances = model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
        indexes.append(distances.argmin(dim=-1).cpu())
    indexes = torch.cat(indexes, dim=0)
    print(f"Exporting indexes with shape {indexes.shape}.")
    torch.save(indexes, export_path)
