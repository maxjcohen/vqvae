from typing import Optional

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Residual connection block, as defined in ResNet.

    Parameters
    ----------
    in_channels : number of input channels.
    out_channels : number of output channels.
    stride : number of stride for convolutions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: Optional[int] = 1
    ) -> None:
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=self._stride,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if self.require_shotcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=self._stride,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs computation of the residual block.

        Parameters
        ----------
        x: input tensor.

        Returns
        -------
        transformed tensor.
        """
        residual = x
        if self.require_shotcut:
            residual = self.shortcut(residual)

        out = self.block(x)
        out = out + residual
        out = nn.ReLU()(out)
        return out

    @property
    def require_shotcut(self) -> None:
        return self._stride > 1 or self._in_channels != self._out_channels
