"""
Code for sampling from diffusion models
"""
from cmath import isnan
import logging
from typing import *

from tqdm.auto import tqdm

import torch
from torch import nn

import beta_schedules
import utils


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, l in enumerate(seq_lens):
        attn_mask[i, :l] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attention_mask=attn_mask)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    noise_modulo: Optional[Union[float, torch.Tensor]] = None,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with modulo {noise_modulo} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        # Shape is (batch, seq_len, 4)
        img = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
    return torch.stack(imgs)


@torch.no_grad()
def sample(
    model: nn.Module,
    seq_lens: Sequence[int],
    seq_max_len: int,
    betas: torch.Tensor,
    posterior_variance: torch.Tensor,
    batch_size: int = 16,
    channels: int = 4,
    timesteps: int = 200,
    noise_modulo: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    retval = p_sample_loop(
        model,
        lengths=seq_lens,
        shape=(batch_size, seq_max_len, channels),
        timesteps=timesteps,
        betas=betas,
        posterior_variance=posterior_variance,
        noise_modulo=noise_modulo,
    )[-1]

    # Trim the sequences by sequence lengths
    retval = [retval[i, :l, :] for i, l in enumerate(seq_lens)]
    return retval
