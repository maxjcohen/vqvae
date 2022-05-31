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
