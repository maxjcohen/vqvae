import torch
import torch.nn.functional as F


@torch.no_grad()
def reconstruction(model, dataloader):
    cost = []
    for images, _ in dataloader:
        reconstructions = model(images)
        cost.append(F.mse_loss(reconstructions, images))
    return torch.Tensor(cost).mean()
