import torch


class Codebook(torch.nn.Embedding):
    def __init__(self, num_codebook, dim_codebook, **kwargs):
        super().__init__(num_embeddings=num_codebook, embedding_dim=dim_codebook)
        self.weight.data.uniform_(-1 / num_codebook, 1 / num_codebook)

    def quantize(self, encoding: torch.Tensor) -> torch.Tensor:
        """Quantize an encoding vector with respect to the codebook.

        Compute the distances between the encoding and the codebook vectors, and assign
        the closest codebook to each point in the feature map.

        Parameters
        ----------
        encoding: input tensor with shape `(*, D)`.

        Returns
        -------
        Quantized tensor with the same shape as the input vector.
        """
        distances = self.compute_distances(encoding)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook_lookup(indices)
        return quantized

    def compute_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        """Compute distance between encodings and codebooks.

        Parameters
        ----------
        encodings: encoding tensor with shape `(*, D)`.

        Returns
        -------
        Distances with shape `(*, K)`.
        """
        distances = (encodings.unsqueeze(-2) - self.weight).square().mean(-1)
        return distances

    def codebook_lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Associate tensor of indices with the corresponding codebooks.

        Parameters
        ----------
        indices: Discrete vector with shape (*).

        Returns
        -------
        Tensor of codebooks with shape `(*, D)`.
        """
        return self(indices)

    @property
    def num_codebook(self):
        return self.num_embeddings

    @property
    def dim_codebook(self):
        return self.embedding_dim
