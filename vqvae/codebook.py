import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(torch.nn.Embedding):
    """VQ-VAE Codebook.

    Implements the algorithm presented in "Neural Discrete Representation Learning"
    [https://arxiv.org/abs/1711.00937]. This module stores a finite set of codebooks,
    used to compute quantization of input tensors.

    Note
    ----
    This module inherits from the `torch.nn.Embedding` module, and uses its
    forward implementation for the `codebook_lookup` function.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of codebooks.
    """

    def __init__(self, num_codebook: int, dim_codebook: int, **kwargs):
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


class EMACodebook(Codebook):
    """VQ-VAE Codebook with Exponential Moving Avergage (EMA).

    This modules is similar to the `Codebook` module, with the addition of updating
    codebook positions with EMA.

    Note
    ----
    Because codebooks are updated directly with EMA, quantized vector do not hold any
    gradient.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of codebooks.
    gamma: degree of weighting decrease. Must be between `0` and `1`. Default is `0.99`.
    epsilon: pseudocount for the Laplace smoothing. Default is `1e-5`.
    """

    def __init__(
        self,
        num_codebook: int,
        dim_codebook: int,
        gamma: float = 0.99,
        epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__(num_codebook=num_codebook, dim_codebook=dim_codebook, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_codebook))
        self.register_buffer("ema_positions", self.weight.clone())

    def quantize(self, encoding: torch.Tensor) -> torch.Tensor:
        original_shape = encoding.shape
        encoding = encoding.reshape(-1, self.dim_codebook)
        distances = self.compute_distances(encoding)
        indices = torch.argmin(distances, dim=-1)
        indices_onehot = F.one_hot(indices, num_classes=self.num_codebook)
        if self.training:
            # Update cluster size
            instant_cluster_size = indices_onehot.sum(dim=0)
            self.ema_cluster_size = (
                self.gamma * self.ema_cluster_size
                + (1 - self.gamma) * instant_cluster_size
            )
            total_cluster_size = self.ema_cluster_size.sum(dim=-1, keepdims=True)
            # Laplace smoothing to avoid empty clusters
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (total_cluster_size + self.num_codebook * self.epsilon)
                * total_cluster_size
            )
            # Update positions
            instant_positions = indices_onehot.T.to(dtype=torch.float32) @ encoding
            self.ema_positions = (
                self.gamma * self.ema_positions + (1 - self.gamma) * instant_positions
            )
            # Update codebook
            self.weight.data = self.ema_positions / self.ema_cluster_size.unsqueeze(-1)
        quantized = self.codebook_lookup(indices).view(original_shape)
        return quantized.detach()
