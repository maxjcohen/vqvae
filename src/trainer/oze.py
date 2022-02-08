import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import plotly.graph_objects as go
from aim import Figure


def aim_fig_plot_ts(arrays):
    fig = go.Figure()
    for name, array in arrays.items():
        fig.add_trace(
            go.Scatter(
                y=array.detach().cpu().numpy().squeeze(), mode="lines", name=name
            )
        )
    fig.update_layout(legend=dict(xanchor="left", x=0.5))
    return Figure(fig)


def flatten_ts_batch(batch):
    return torch.cat([sample for sample in batch.transpose(0, 1)], dim=0)


class LitOzeTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = model

    def training_step(self, batch, batch_idx):
        u, y = batch
        encoding = self.model.encode(y)
        quantized, codebook_metrics = self.model.codebook.quantize(encoding)
        reconstructions = self.model.decode(quantized)
        loss = F.mse_loss(reconstructions, y) + codebook_metrics["loss_latent"]
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "perplexity", codebook_metrics["perplexity"], on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        u, y = batch
        reconstructions = self.model(y)
        reconstruction_loss = F.mse_loss(reconstructions, y)
        self.log("val_reconstruction_loss", reconstruction_loss)
        if batch_idx == 0:
            self.logger.experiment.track(
                aim_fig_plot_ts(
                    {"observations": y[:, 1], "predictions": reconstructions[:, 1]}
                ),
                name="batch-comparison",
                epoch=self.current_epoch,
                context={"subset": "val"},
            )
        return y, reconstructions

    def validation_epoch_end(self, outputs):
        originals = torch.cat([flatten_ts_batch(batch) for batch, _ in outputs], dim=0)
        predictions = torch.cat(
            [flatten_ts_batch(batch) for _, batch in outputs], dim=0
        )
        self.logger.experiment.track(
            aim_fig_plot_ts({"observations": originals, "predictions": predictions}),
            name="full-comparison",
            epoch=self.current_epoch,
            context={"subset": "val"},
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)
