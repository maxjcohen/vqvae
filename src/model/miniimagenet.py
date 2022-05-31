from typing import Optional

from vqvae.codebook import Codebook, EMACodebook, GumbelCodebook
from .cifar import CifarVQVAE
from .backbone import MiniimagenetAutoEncoder


class MiniImagenetVQVAE(CifarVQVAE):
    def __init__(
        self,
        num_codebook: int,
        dim_codebook: int,
        codebook_flavor: Optional[str] = "ema",
    ):
        super().__init__(num_codebook=num_codebook, dim_codebook=dim_codebook)
        self.autoencoder = MiniimagenetAutoEncoder(out_channels=dim_codebook)
        self.encode = self.autoencoder.encode
        self.decode = self.autoencoder.decode
