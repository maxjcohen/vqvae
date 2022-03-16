from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


class Codebook(torch.nn.Embedding):
    """VQ-VAE Codebook.

    Implements the algorithm presented in "Neural Discrete Representation Learning"
    [https://arxiv.org/abs/1711.00937], corresponding to the distribution:

    ..math
        q_\varphi(z_q = e_k | z_e) = \delta_k ( \argmin_{l=1}^K \| z_e - e_l \| )
        \quad \forall 1 \leq k \leq K

    This module stores a finite set of codebooks, used to compute quantization of input tensors.

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
        self._eps = torch.finfo(torch.float32).eps

    def quantize(self, encoding: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Quantize an encoding vector with respect to the codebook.

        Compute the distances between the encoding and the codebook vectors, and assign
        the closest codebook to each point in the feature map. The gradient from the
        return quantized tensor is copied to the encoding. This function also returns
        the latent loss and the normalized perplexity.

        Parameters
        ----------
        encoding: input tensor with shape `(*, D)`.

        Returns
        -------
        quantized: quantized tensor with the same shape as the input vector.
        indices: indices sampled with shape `(*, K)`.
        losses: dictionary containing the latent loss between encodings and selected
        codebooks, and the normalized perplexity.
        """
        distances = self.compute_distances(encoding)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook_lookup(indices)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        # Compute perplexity
        indices_onehot = F.one_hot(indices, num_classes=self.num_codebook)
        probs = (
            indices_onehot
            .view(-1, self.num_codebook)
            .float()
            .mean(dim=0)
        )
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + self._eps)))
        perplexity = perplexity / self.num_codebook
        return quantized, indices_onehot, {
            "loss_latent": loss_latent,
            "perplexity": perplexity,
        }

    def compute_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        """Compute distance between encodings and codebooks.

        Parameters
        ----------
        encodings: encoding tensor with shape `(*, D)`.

        Returns
        -------
        Distances with shape `(*, K)`.
        """
        distances = (encodings.unsqueeze(-2) - self.weight).square().sum(-1)
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

    def quantize(self, encoding: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Quantize an encoding vector with respect to the codebook.

        Note
        ----
        Because codebooks are updated using EMA, the latent loss does not back propagate
        gradient toward the codebooks.

        Parameters
        ----------
        encoding: input tensor with shape `(*, D)`.

        Returns
        -------
        quantized: quantized tensor with the same shape as the input vector.
        indices: indices sampled with shape `(*, K)`.
        losses: dictionary containing the latent loss between encodings and selected
        codebooks, and the normalized perplexity.
        """
        encoding_flatten = encoding.reshape(-1, self.dim_codebook)
        distances = self.compute_distances(encoding_flatten)
        indices = torch.argmin(distances, dim=-1)
        indices_onehot = F.one_hot(indices, num_classes=self.num_codebook)
        probs = indices_onehot.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + self._eps)))
        perplexity = perplexity / self.num_codebook
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
            instant_positions = indices_onehot.T.float() @ encoding_flatten
            self.ema_positions = (
                self.gamma * self.ema_positions + (1 - self.gamma) * instant_positions
            )
            # Update codebook
            self.weight.data = self.ema_positions / self.ema_cluster_size.unsqueeze(-1)
        quantized = self.codebook_lookup(indices).view(encoding.shape)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        return quantized, indices_onehot, {
            "loss_latent": loss_latent,
            "perplexity": perplexity,
        }


class GumbelCodebook(Codebook):
    """VQ-VAE Codebook with Gumbel Sotmax Reparametrization.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of codebooks.
    tau: temperature for the Gumbel Softmax distribution. Default is `0.5`.
    """

    def __init__(self, num_codebook: int, dim_codebook: int, tau: float = 0.5):
        super().__init__(num_codebook=num_codebook, dim_codebook=dim_codebook)
        self.tau = tau

    def quantize(self, encoding: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Quantize the encoded vector using Gumbel Softmax.

        Implements a quantization based on the distribution:
        ..math
            q_\varphi(z_q = e_k | z_e) = \frac{\exp{ - \| z_e - e_k \|^2 }}
            {\sum_{l=1}^K \exp { - \| z_e - e_l \|^2 }}
            \quad \forall 1 \leq k \leq K

        Note
        ----
        During training, we sample instead from the associated Gumbel Softmax
        distribution in order to allow gradient propagation.

        Returns
        -------
        quantized: quantized tensor with the same shape as the input vector.
        losses: dictionary containing the latent loss between encodings and selected
        codebooks, and the normalized perplexity.
        """
        encoding_flatten = encoding.reshape(-1, self.dim_codebook)
        distances = self.compute_distances(encoding_flatten)
        gumbel_softmax = RelaxedOneHotCategorical(
            logits=-distances, temperature=self.tau
        )
        indices = gumbel_softmax.rsample()
        if self.training:
            quantized = indices @ self.weight
        else:
            quantized = self.codebook_lookup(indices.argmax(-1))
        # KL divergence
        kl = gumbel_softmax.probs * torch.log(
            gumbel_softmax.probs * self.num_codebook + self._eps
        )
        kl = kl.sum()
        # Compute perplexity
        probs = indices.mean(dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + self._eps)))
        perplexity = perplexity / self.num_codebook
        quantized = quantized.view(encoding.shape)
        indices = indices.view(*encoding.shape[:-1], -1)
        return quantized, indices, {"loss_latent": kl, "perplexity": perplexity}
