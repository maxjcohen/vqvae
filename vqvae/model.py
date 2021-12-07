from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .modules import ResNetBlock
from .codebook import Codebook


class VQVAE(nn.Module):
    """VQVAE model based on a ResNet architecture.

    The input vectors are first encoded, then quantize using a codebook, and finally
    decoded going through the same dimensions as the encoder.

    Example
    -------
        n_codebook = 128
        dim_codebook = 32
        channel_sizes = [16, 64, dim_codebook]
        strides = [2, 2, 1]
        MNISTModel(in_channel=3, channel_sizes, n_codebook, dim_codebook, strides)

    Note
    ----
    In the following documentation, we will use the following variable names:
    `B`: batch size.
    `W` and `H`: width and height of images or feature map.
    `C`: number of channels of the final encoding layers. This must be equal to the
    dimension of the codebooks.
    `N`: number of codebooks.
    """

    def __init__(
        self,
        in_channel: int,
        channel_sizes: List[int],
        n_codebook,
        dim_codebook,
        strides: Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        in_channel: number of input channels.
        channel_sizes: sequence of channel sizes of the encoder and decoder layers.
        strides: sequence of strides, default is 1 for every layer.
        """
        super().__init__()

        strides = strides or [1 for _ in channel_sizes]
        assert len(strides) == len(channel_sizes)
        assert channel_sizes[-1] == dim_codebook

        channel_sizes = [in_channel, *channel_sizes]
        self.encoder = torch.nn.ModuleList(
            [
                ResNetBlock(in_channel, out_channel, stride=stride)
                for in_channel, out_channel, stride in zip(
                    channel_sizes[:-1], channel_sizes[1:], strides
                )
            ]
        )
        self.decoder = torch.nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channel,
                    out_channels=in_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
                for in_channel, out_channel, stride in reversed(
                    list(zip(channel_sizes[:-1], channel_sizes[1:], strides))
                )
            ]
        )

        self.decoder_activation = nn.ReLU()
        self.emb = Codebook(n_codebook, dim_codebook)

        self._dim_codebook = dim_codebook

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding of the input tensor.

        Parameters
        ----------
        x: input tensor with shape `(B, *, *, *)`.

        Returns
        -------
        encoded vector with shape `(B, C, W, H)`.
        """
        self.encoder_shapes = []

        for layer in self.encoder:
            self.encoder_shapes.append(x.shape)
            x = layer(x)
        return x

    def quantize(self, encoding: torch.Tensor) -> torch.Tensor:
        """Quantize an encoding vector with respect to the codebook.

        Compute the distances between the encoding and the codebook vectors, and assign
        the closest codebook to each point in the feature map.

        Parameters
        ----------
        encoding: input tensor with shape `(B, C, W, H)`.

        Returns
        -------
        Encoding tensor with shape `(B, C, W, H)`.
        """
        distances = self.emb.compute_distances(encoding.permute(0, 2, 3, 1))
        quantized = torch.argmin(distances, dim=-1)
        encodings = self.emb.codebook_lookup(quantized)
        encodings = encodings.permute(0, 3, 1, 2)
        return encodings

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the decoding of the input encoded vector.

        Parameters
        ----------
        x: encoded vector with shape `(B, C, W, H)`.

        Returns
        -------
        decoded vector with shape `(B, *, *, *)` matching the input vector of the
        encoding process.
        """
        for idx, (layer, output_shape) in enumerate(
            zip(self.decoder, reversed(self.encoder_shapes))
        ):
            x = layer(x, output_size=output_shape)
            if idx < len(self.encoder_shapes) - 1:
                x = self.decoder_activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder, quantize and decode.

        Parameters
        ----------
        x: input tensor with shape `(B, *, *, *)`.

        Returns
        -------
        decoded representation with equivalent shape `(B, *, *, *)`.
        """
        encodings = self.encode(x)
        qt = self.quantize(encodings)
        return self.decode(qt)

    @property
    def dim_codebook(self):
        return self._dim_codebook
