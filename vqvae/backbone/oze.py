import torch
import torch.nn as nn


class OzeBackbone(nn.Module):
    """Oze Backbone.

    Note
    ----
    In the following documentation, we will use the following variable names:
    `B`: batch size.
    `D`: number of channels of the feature map.

    Parameters
    ----------
    latent_dim: number of channels of the feature map.
    """

    _input_dim = 1
    _output_dim = 1
    _num_layers = 2

    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=self._input_dim,
            hidden_size=latent_dim,
            num_layers=self._num_layers,
            dropout=0.1,
        )

        self.decoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=self._num_layers,
            dropout=0.1,
        )
        self.output_mean = nn.Linear(latent_dim, self._output_dim)
        self.output_logvar = nn.Linear(latent_dim, self._output_dim)

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, x):
        decoded_latent = self.decoder(x)[0]
        return (
            self.output_mean(decoded_latent),
            self.output_logvar(decoded_latent).exp(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder and the decoder.

        Parameters
        ----------
        x: input tensor with shape `(T, B, 2)`.

        Returns
        -------
        Decoded representation with shape `(T, B, 1)`
        """
        return self.decode(self.encode(x))
