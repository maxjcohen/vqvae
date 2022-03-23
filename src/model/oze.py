from typing import Optional

import torch
import torch.nn as nn

from vqvae.codebook import Codebook, EMACodebook, GumbelCodebook
from .backbone import OzeBackbone


class OzeVQVAE(nn.Module):
    """VQVAE model for the Oze dataset.

    This module combines a backbone for the Oze dataset with a vqvae codebook.

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
        self.backbone = OzeBackbone(latent_dim=dim_codebook)
        self.encode = self.backbone.encode
        self.decode = self.backbone.decode
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
        x: input tensor with shape `(T, B, 2)`.

        Returns
        -------
        Decoded representation with shape `(T, B, 1)`.
        """
        encoding = self.encode(x)
        quantized = self.codebook.quantize(encoding)[0]
        return self.decode(quantized)
