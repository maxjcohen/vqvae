import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def reconstruction(model, dataloader, device='cpu'):
    cost = []
    for images, _ in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        reconstructions = torch.sigmoid(model(images))
        cost.append(F.mse_loss(reconstructions, images))
    return torch.Tensor(cost).mean()


@torch.no_grad()
def rank(model, dataloader):
    rank = []
    for images, _ in tqdm(dataloader, total=len(dataloader)):
        encodings = model.encode(images)
        qt = model.quantize(encodings)
        n_uniques = len(
            qt.permute(0, 2, 3, 1).view(-1, model.emb.embedding_dim).unique(dim=0)
        )
        rank.append(n_uniques)
    return torch.Tensor(rank).mean() / model.emb.num_embeddings
