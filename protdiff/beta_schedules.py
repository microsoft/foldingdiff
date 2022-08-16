"""
Describe beta schedules
"""
import os
import logging
from typing import Dict, Literal, get_args

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F

SCHEDULES = Literal["linear", "cosine", "quadratic"]


# each of these returns the series of beta_t


def cosine_beta_schedule(timesteps: int, s: float = 8e-3) -> torch.Tensor:
    """
    Cosine scheduling https://arxiv.org/pdf/2102.09672.pdf
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


def compute_alphas(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute the alphas from the betas
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def get_variance_schedule(keyword: SCHEDULES, timesteps: int, **kwargs) -> torch.Tensor:
    """
    Easy interface for getting a variance schedule based on keyword and
    number of timesteps
    """
    logging.info(f"Getting {keyword} variance schedule with {timesteps} timesteps")
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
    )
    logging.info(
        f"Plotting {keyword} variance schedule with {timesteps} timesteps, ranging from {torch.min(variance_vals)}-{torch.max(variance_vals)}"
    )
    alpha_beta_vals = compute_alphas(variance_vals)
    fig, ax = plt.subplots(dpi=300)
    for k, v in alpha_beta_vals.items():
        ax.plot(np.arange(timesteps), v.numpy(), label=k, alpha=0.7)
    ax.legend()
    ax.set(
        title=f"{keyword} schedule across {timesteps} timesteps",
        xlabel="Timestep",
    )
    fig.savefig(fname)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from plotting import PLOT_DIR

    var_plot_dir = os.path.join(PLOT_DIR, "variance_schedules")
    if not os.path.isdir(var_plot_dir):
        os.makedirs(var_plot_dir)
    for s in get_args(SCHEDULES):
        plot_variance_schedule(
            os.path.join(var_plot_dir, f"{s}_var_schedule.pdf"), keyword=s
        )
