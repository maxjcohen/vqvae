from typing import Optional

import torch
import torch.nn as nn

from .codebook import Codebook, EMACodebook, GumbelCodebook


class GenericVQVAE(nn.Module):
    """Generic VQVAE model.

    This module combines an generic autoencoder with a vqvae codebook.

    Parameters
    ----------
    num_codebook: number of codebooks.
    dim_codebook: dimension of each codebook vector.
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
        CodebookFlavor = {
            "classic": Codebook,
            "ema": EMACodebook,
            "gumbel": GumbelCodebook,
        }[codebook_flavor]
        self.codebook = CodebookFlavor(num_codebook, dim_codebook)
        self.codebook_flavor = codebook_flavor

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder, quantize and decode.

        Parameters
        ----------
        x: input tensor.

        Returns
        -------
        Decoded representation with the same shape as the input tensor.
        """
        encoding = self.encode(x)
        quantized = self.codebook.quantize(encoding)[0]
        return self.decode(quantized)
