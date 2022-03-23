from typing import Optional

import torch
import torch.nn as nn

from vqvae.codebook import Codebook, EMACodebook, GumbelCodebook
from .backbone import CifarAutoEncoder


class CifarVQVAE(nn.Module):
    """VQVAE model for the CIFAR dataset.

    This module combines an autoencoder for the CIFAR dataset with a vqvae codebook.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of each codebook vector. This value will set the number of
    output channels of the encoder.
    codebook_flavor: one of `"classic"` (for basic vqvae), `"ema"` (for Exponential
    Moving Average (EMA)) or `"gumbel"` (for GumbelSoftmax quantization). Default is
    `"ema"`.
    """

    def __init__(
        self,
        num_codebook: int,
        dim_codebook: int,
        codebook_flavor: Optional[str] = "ema",
    ):
        super().__init__()
        self.autoencoder = CifarAutoEncoder(out_channels=dim_codebook)
        self.encode = self.autoencoder.encode
        self.decode = self.autoencoder.decode
        CodebookFlavor = {
            "classic": Codebook,
            "ema": EMACodebook,
            "gumbel": GumbelCodebook,
        }[codebook_flavor]
        self.codebook = CodebookFlavor(num_codebook, dim_codebook)
        self.codebook_flavor = codebook_flavor

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
        quantized = self.codebook.quantize(encoding)[0]
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        return self.decode(quantized)
