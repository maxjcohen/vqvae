import torch
from tqdm import tqdm


@torch.no_grad()
def export_distances(model, dataloader, reduction="mean"):
    for images, _ in tqdm(dataloader, total=len(dataloader)):
        encodings = model.encode(images)
        distances = model.emb.compute_distances(
            encodings.permute(0, 2, 3, 1), reduction=reduction
        )
        print(distances.shape)
