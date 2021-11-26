import torch
from tqdm import tqdm


@torch.no_grad()
def export_distances(model, dataloader, reduction="mean"):
    distances = []
    for images, _ in tqdm(dataloader, total=len(dataloader)):
        encodings = model.encode(images)
        distances.append(
            model.emb.compute_distances(
                encodings.permute(0, 2, 3, 1), reduction=reduction
            )
        )
    distances = torch.cat(distances, dim=0)
    return distances
