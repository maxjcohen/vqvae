from pathlib import Path

import torch
from tqdm import tqdm


@torch.no_grad()
def export_distances(model, dataloader, export_path="."):
    export_path = Path(export_path)
    export_path.mkdir(exist_ok=True)
    for idx_batch, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        encodings = model.encode(images)
        distances = model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
        torch.save(distances, export_path / f"batch_{idx_batch}.pt")
