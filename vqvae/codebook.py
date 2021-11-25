import torch


class Codebook(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def compute_distances(self, encodings, reduction="mean"):
        """
        reduction: mean | none
        encodings: (B, W, H, C)

        returns
        distances (B, W, H, N) if "mean" else (B, W, H, C, N)
        """
        distances = (encodings.unsqueeze(-2) - self.weight).square()
        if reduction == "mean":
            distances = distances.mean(-1)
        return distances

    @staticmethod
    def quantize(distances):
        """
        distances: (B, W, H, N)

        returns
        indexes: (B, W, H)
        """
        return torch.argmin(distances, dim=-1)

    def codebook_lookup(self, qt):
        """
        qt: (B, W, H)

        returns
        zq: (B, W, H, C)

        """
        return self(qt)
