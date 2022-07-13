"""
Describe beta schedules
"""
import os
from typing import Literal, get_args

import numpy as np
from matplotlib import pyplot as plt

import torch

SCHEDULES = Literal["linear", "cosine", "quadratic"]


def cosine_beta_schedule(timesteps: int, s: float = 8e-3) -> torch.Tensor:
    """
    Cosine scheduling
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int, beta_start=1e-4, beta_end=0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    timesteps: int, beta_start=1e-4, beta_end=0.02
) -> torch.Tensor:
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_variance_schedule(keyword: SCHEDULES, timesteps: int, **kwargs) -> torch.Tensor:
    """
    Easy interface for getting a variance schedule based on keyword and
    number of timesteps
    """
    if keyword == "cosine":
        return cosine_beta_schedule(timesteps, **kwargs)
    elif keyword == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    elif keyword == "quadratic":
        return quadratic_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"Unrecognized variance schedule: {keyword}")


def plot_variance_schedule(
    fname: str, keyword: SCHEDULES, timesteps: int = 1000, **kwargs
):
    """
    Plot the given variance schedule
    """
    variance_vals = get_variance_schedule(
        keyword=keyword, timesteps=timesteps, **kwargs
    ).numpy()

    fig, ax = plt.subplots(dpi=300)
    ax.plot(np.arange(timesteps), variance_vals)
    fig.savefig(fname)


if __name__ == "__main__":
    from plotting import PLOT_DIR

    var_plot_dir = os.path.join(PLOT_DIR, "variance_schedules")
    if not os.path.isdir(var_plot_dir):
        os.makedirs(var_plot_dir)
    for s in get_args(SCHEDULES):
        plot_variance_schedule(
            os.path.join(var_plot_dir, f"{s}_var_schedule.pdf"), keyword=s
        )
