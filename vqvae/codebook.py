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

    This module stores a finite set of codebooks, used to compute quantization of input
    tensors.

    Note
    ----
    This module inherits from the `torch.nn.Embedding` module, and uses its forward
    implementation for the `codebook_lookup` function.

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
        losses: dictionary containing the latent loss between encodings and selected
        codebooks, and the normalized perplexity.
        """
        distances = self.compute_distances(encoding)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook_lookup(indices)
        loss_latent = F.mse_loss(encoding, quantized)
        quantized = encoding + (quantized - encoding).detach()
        # Compute perplexity
        probs = (
            F.one_hot(indices, num_classes=self.num_codebook)
            .view(-1, self.num_codebook)
            .float()
            .mean(dim=0)
        )
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + self._eps)))
        perplexity = perplexity / self.num_codebook
        return quantized, {
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
        return quantized, {
            "loss_latent": loss_latent,
            "perplexity": perplexity,
        }


class GumbelCodebook(Codebook):
    """VQ-VAE Codebook with Gumbel Sotmax Reparametrization.

    Implements a quantization based on the distribution:
    ..math
        q_\varphi(z_q = e_k | z_e) = \frac{\exp{ - \| z_e - e_k \|^2 }}
        {\sum_{l=1}^K \exp { - \| z_e - e_l \|^2 }}
        \quad \forall 1 \leq k \leq K

    Note
    ----
    During training, we sample instead from the associated Gumbel Softmax distribution
    in order to allow gradient propagation.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of codebooks.
    tau: temperature for the Gumbel Softmax distribution. Default is `0.5`.
    """

    def __init__(self, num_codebook: int, dim_codebook: int, tau: float = 0.5):
        super().__init__(num_codebook=num_codebook, dim_codebook=dim_codebook)
        self.tau = tau

    def quantize(
        self, encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Quantize the encoded vector using Gumbel Softmax.

        Parameters
        ----------
        encoding: encoding tensor with shape `(*, D)`.

        Returns
        -------
        quantized: quantized tensor with the same shape as the input vector.
        sample: sampled soft indices with shape `(*, K)`.
        losses: dictionary containing the latent loss between encodings and selected
        codebooks, and the normalized perplexity.
        """
        encoding_flatten = encoding.reshape(-1, self.dim_codebook)
        quantized, sample, logits = self.sample(encoding_flatten)
        loss_posterior = self.loss_posterior(sample, logits)
        perplexity = self.perplexity(sample)
        return (
            quantized.view(encoding.shape),
            sample.view(*encoding.shape[:-1], -1),
            {"loss_latent": loss_posterior, "perplexity": perplexity},
        )

    def sample(
        self, encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a quantized vector associated with an encoding.

        During training, soft indices are sampled from a Gumbel Softmax distribution,
        the resulting quantized vector is a linear combination of the codebooks. During
        evaluation, the quantized vector is sampled as from a traditional Categorical
        distribution.

        Parameters
        ----------
        encoding: encoding tensor with shape `(B, D)`.

        Returns
        -------
        quantized: quantized tensor with the same shape as the input vector.
        sample: sampled soft indices with shape `(B, K)`.
        logits: unnormalized log probabilities with shape `(B, K)`.
        """
        logits = -self.compute_distances(encoding)
        gumbel_softmax = RelaxedOneHotCategorical(logits=logits, temperature=self.tau)
        sample = gumbel_softmax.rsample()
        if self.training:
            quantized = sample @ self.weight
        else:
            quantized = self.codebook_lookup(sample.argmax(-1))
        return quantized, sample, logits

    def loss_posterior(self, sample: torch.Tensor, logits: torch.Tensor) -> torch.float:
        """Compute the posterior term of the ELBO.

        We assume the `sample` tensor in sampled from a Soft gumbel Softmax
        distribution:

        ..math
            z_q = \sum_{k=1}^K q_k e_k

        We approximate the expectation, with respect to q, to q, by using this single
        sample:

        ..math
            \mathbb{E}_{q_\varphi} [\log q_\varphi] =
            \sum_{k=1}^K q_k \log q_\varphi(z_q)

        Parameters
        ----------
        sample: sample with shape `(B, K)`.
        logits: unnormalized log probabilities with shape `(B, K)`.
        """
        return -F.cross_entropy(logits, sample)

    def perplexity(self, sample: torch.Tensor) -> torch.float:
        """Compute the perplexity associated with the given sample.

        Parameters
        ----------
        sample: sample with shape `(B, K)`.
        """
        probs = sample.mean(dim=0).detach()
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + self._eps)))
        perplexity = perplexity / self.num_codebook
        return perplexity
