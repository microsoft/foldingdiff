"""
Code for sampling from diffusion models
"""
from typing import *

from tqdm.auto import tqdm

import torch

import utils


@torch.no_grad()
def p_sample(model, x, t, t_index, betas, posterior_variance):
    # Calculate alphas and betas
    alphas = 1.0 - betas
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    betas_t = utils.extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = utils.extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = utils.extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(
    model, shape: Tuple[int], timesteps: int, betas, posterior_variance
) -> "list[torch.Tensor]":
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        img = p_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            i,
            betas=betas,
            posterior_variance=posterior_variance,
        )
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample(
    model,
    seq_len,
    betas,
    posterior_variance,
    batch_size=16,
    channels=4,
    timesteps: int = 200,
) -> "list[torch.Tensor]":
    return p_sample_loop(
        model,
        shape=(batch_size, seq_len, channels),
        timesteps=timesteps,
        betas=betas,
        posterior_variance=posterior_variance,
    )
