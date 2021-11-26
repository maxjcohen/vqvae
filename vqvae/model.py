from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .modules import ResNetBlock
from .codebook import Codebook


class VQVAE(nn.Module):
    """ResNet model for the MNIST dataset.

    This model contains an encoder model, based on ResNet blocks, and a deconvolution
    decoder going through the same dimensions as the encoder.

    Example
    -------
        channel_sizes = [16, 64, 128]
        strides = [2, 3, 3]
        MNISTModel(in_channel=3, channel_sizes=channel_sizes, strides=strides)
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

        # channel_sizes.insert(0, in_channel)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding of the input tensor, compute distance with codebook, quanticize.

        Intermediate results distances and qt are stored.

        Parameters
        ----------
        x: input tensor.

        Returns
        -------
        encoded vector.
        """
        self.encoder_shapes = []

        for layer in self.encoder:
            self.encoder_shapes.append(x.shape)
            x = layer(x)
        return x

    def quanticize(self, encoding):
        self.distances = self.emb.compute_distances(encoding.permute(0, 2, 3, 1))
        self.qt = self.emb.quantize(self.distances)
        vq = self.emb.codebook_lookup(self.qt)
        vq = vq.permute(0, 3, 1, 2)
        return vq

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the decoding of the input encoded vector.

        Parameters
        ----------
        x: encoded vector.

        Returns
        -------
        decoded vector.
        """
        for idx, (layer, output_shape) in enumerate(
            zip(self.decoder, reversed(self.encoder_shapes))
        ):
            x = layer(x, output_size=output_shape)
            if idx < len(self.encoder_shapes) - 1:
                x = self.decoder_activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder and decoder.

        Parameters
        ----------
        x: input tensor.

        Returns
        -------
        decoded representation.
        """
        encodings = self.encode(x)
        qt = self.quanticize(encodings)
        return self.decode(qt)
