import torch


class Codebook(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def compute_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        """Compute distance between encodings and codebooks.

        Parameters
        ----------
        encodings: encoding tensor with shape `(B, W, H, C)`.

        Returns
        -------
        distances with shape `(B, W, H, N)`.
        """
        distances = (encodings.unsqueeze(-2) - self.weight).square().mean(-1)
        return distances

    def codebook_lookup(self, quantized: torch.Tensor) -> torch.Tensor:
        """Associate vectors in the quantized tensor with the corresponding codebook.

        Parameters
        ----------
        quantized: Discrete vector with shape (B, *).

        Returns
        -------
        Tensor of codebooks with shape `(B, *)`.
        """
        return self(quantized)
