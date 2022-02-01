import torch
import torch.nn as nn

from .codebook import Codebook, EMACodebook
from .autoencoder import CifarAutoEncoder


class CifarVQVAE(nn.Module):
    """VQVAE model for the CIFAR dataset.

    This module combines an autoencoder for the CIFAR dataset with a vqvae codebook.
    """

    def __init__(self, num_codebook: int, dim_codebook: int, ema:bool=True):
        """
        Parameters
        ----------
        num_codebook: number of codebooks.
        dim_codebook: dimension of each codebook vector. This value will set the number
        of output channels of the encoder.
        ema: If `True`, codebooks are updated using exponential moving average. Default
        is `True`.
        """
        super().__init__()
        self.autoencoder = CifarAutoEncoder(out_channels=dim_codebook)
        self.encode = self.autoencoder.encode
        self.decode = self.autoencoder.decode
        Codebook = EMACodebook if ema else Codebook
        self.codebook = Codebook(num_codebook, dim_codebook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder, quantize and decode.

        Parameters
        ----------
        x: input tensor with shape `(B, 3, 32, 32)`.

        Returns
        -------
        Decoded representation with the same shape as the input tensor.
        """
        encoding = self.encode(x)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized = self.codebook.quantize(encoding)
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        return self.decode(quantized)
