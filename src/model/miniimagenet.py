from typing import Optional

import torch

from vqvae.codebook import Codebook, EMACodebook, GumbelCodebook
from .backbone import MiniimagenetAutoEncoder


class MiniImagenetVQVAE(torch.nn.Module):
    def __init__(
        self,
        num_codebook: int,
        dim_codebook: int,
        codebook_flavor: Optional[str] = "ema",
    ):
        super().__init__()
        self.autoencoder = MiniimagenetAutoEncoder(out_channels=dim_codebook)
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
