import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def plot_predictions(
    model: callable,
    dataset: torch.utils.data.dataset.Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Plot predictions on the Oze Dataset.

    For every targeted variable, plot prediction against ground truth.

    Parameters
    ----------
    model: prediction model.
    dataset: instance of Oze Dataset.
    batch_size: batch size to iterate over the dataset.
    num_workers: num workers to iterate over the dataset.
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # Load target plots from dataset
    targets = dataset.y
    d_out = targets.shape[-1]

    # Compute predictions from model
    predictions = []
    for inputs, _ in dataloader:
        inputs = inputs.transpose(0, 1)
        predictions.append(model(inputs))
    predictions = torch.cat(predictions, dim=1)
    predictions = predictions.transpose(0, 1).reshape(-1, d_out)

    # Rescale shit
    targets = dataloader.dataset.rescale(targets, "observation")
    predictions = dataloader.dataset.rescale(predictions, "observation")

    # Plot shit
    _, axes = plt.subplots(
        d_out, 1, sharex=True, squeeze=False, figsize=(25, 5 * d_out)
    )
    for target, prediction, ax in zip(targets.T, predictions.T, axes[:, 0]):
        ax.plot(target.numpy(), label="observations", lw=3)
        ax.plot(prediction.numpy(), label="predictions")
        ax.legend()


def compute_occupancy(
    date: datetime, delta: float = 15.0, talon: float = 0.2
) -> np.ndarray:
    if date.weekday() > 4:
        return 0
    if date.hour < 8 or date.hour > 18:
        return 0

    date_start_lockdown = datetime.date(year=2020, month=3, day=17)
    date_end_lockdown = datetime.date(year=2020, month=5, day=11)

    # Full occupancy before lockdown
    occupancy = int(date < date_start_lockdown)

    # After lockdown
    if date_end_lockdown < date:
        # Ocupancy increases gradually
        occupancy += 1 - np.exp(-(date.date() - date_end_lockdown).days / delta)

        # Fixed redduction
        occupancy -= occupancy * talon

    return occupancy
