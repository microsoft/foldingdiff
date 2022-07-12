"""
Describe beta schedules
"""
from typing import Literal
import torch

SCHEDULES = Literal["linear", "cosine", "quadratic"]


def cosine_beta_schedule(timesteps: int, s: float = 8e-3):
    """
    Cosine scheduling
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
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
