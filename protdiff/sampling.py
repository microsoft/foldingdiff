"""
Code for sampling from diffusion models
"""
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn

import beta_schedules
import utils
from datasets import NoisedAnglesDataset


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
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
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


def sample(
    model: nn.Module,
    train_dset: NoisedAnglesDataset,
    n: int,
    sweep_lengths: Optional[Tuple[int, int]] = None,
    batch_size: int = 512,
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        logging.warning(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches")
    retval = []
    for this_lengths in lengths_chunkified:
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        noise = train_dset.sample_noise(
            torch.zeros((batch, train_dset.pad, model.n_inputs), dtype=torch.float32)
        )
        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled = p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular["angles"],
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.means}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular["angles"])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval
