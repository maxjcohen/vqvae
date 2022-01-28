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
        num_codebook = 128
        dim_codebook = 32
        channel_sizes = [16, 64, dim_codebook]
        strides = [2, 2, 1]
        VQVAE(num_codebook, dim_codebook, 3, channel_sizes, strides)

    Note
    ----
    In the following documentation, we will use the following variable names:
    `B`: batch size.
    `W` and `H`: width and height of images or feature map.
    `D`: number of channels of the final encoding layers. This must be equal to the
    dimension of the codebooks.
    `K`: number of codebooks.
    """

    def __init__(
        self,
        num_codebook: int,
        dim_codebook: int,
        in_channel: int,
        channel_sizes: List[int],
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
        self.codebook = Codebook(num_codebook, dim_codebook)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding of the input tensor.

        Parameters
        ----------
        x: input tensor with shape `(B, *, *, *)`.

        Returns
        -------
        Encoded vector with shape `(B, D, W, H)`.
        """
        self.encoder_shapes = []

        for layer in self.encoder:
            self.encoder_shapes.append(x.shape)
            x = layer(x)
        return x

    def decode(self, encoding: torch.Tensor) -> torch.Tensor:
        """Compute the decoding of the input encoded vector.

        Parameters
        ----------
        x: encoded vector with shape `(B, D, W, H)`.

        Returns
        -------
        decoded vector with shape `(B, *, *, *)` matching the input vector of the
        encoding process.
        """
        for idx, (layer, output_shape) in enumerate(
            zip(self.decoder, reversed(self.encoder_shapes))
        ):
            encoding = layer(encoding, output_size=output_shape)
            if idx < len(self.encoder_shapes) - 1:
                encoding = self.decoder_activation(encoding)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder, quantize and decode.

        Parameters
        ----------
        x: input tensor with shape `(B, *, *, *)`.

        Returns
        -------
        decoded representation with equivalent shape `(B, *, *, *)`.
        """
        encoding = self.encode(x)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized = self.codebook.quantize(encoding)
        # Switch to channel first
        quantized = quantized.permute(0, 3, 1, 2)
        return self.decode(quantized)
