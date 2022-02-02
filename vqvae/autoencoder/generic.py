from typing import List, Optional

import torch
import torch.nn as nn

from ..modules import ResNetBlock


class GenericAutoEncoder(nn.Module):
    """Generic AutoEncoder architecture based on ResNet blocks.

    Example
    -------
        channel_sizes = [16, 64, 128]
        strides = [2, 2, 1]
        GenericAutoEncoder(3, channel_sizes, strides)

    Note
    ----
    In the following documentation, we will use the following variable names:
    `B`: batch size.
    `W_0` and `H_0`: width and height of the input images.
    `C`: number of channels of the input images.
    `W` and `H`: width and height of the feature map.
    `D`: number of channels of the feature map.

    Parameters
    ----------
    in_channel: number of input channels.
    channel_sizes: sequence of channel sizes of the encoder and decoder layers.
    strides: sequence of strides, default is 1 for every layer.
    """

    def __init__(
        self,
        in_channel: int,
        channel_sizes: List[int],
        strides: Optional[List[int]] = None,
    ):
        super().__init__()
        strides = strides or [1 for _ in channel_sizes]
        assert len(strides) == len(channel_sizes)

        channel_sizes = [in_channel, *channel_sizes]
        self.encoder = nn.ModuleList(
            [
                ResNetBlock(in_channel, out_channel, stride=stride)
                for in_channel, out_channel, stride in zip(
                    channel_sizes[:-1], channel_sizes[1:], strides
                )
            ]
        )
        self.decoder = nn.ModuleList(
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor.

        Parameters
        ----------
        x: input tensor with shape `(B, C, W_0, H_0)`.

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
        """Decode the input encoded vector.

        Parameters
        ----------
        encoding: encoded vector with shape `(B, D, W, H)`.

        Returns
        -------
        decoded vector with shape `(B, C, W_0, H_0)` matching the input vector of the
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
        """Propagate the input tensor through the encoder and the decoder.

        Parameters
        ----------
        x: input tensor with shape `(B, C, W_0, H_0)`.

        Returns
        -------
        Decoded representation with equivalent shape `(B, C, W_0, H_0)`.
        """
        return self.decode(self.encode(x))
