"""
Script to sample from a trained diffusion model
"""
import os, sys
import logging
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

import torch
from torch.nn import functional as F

from transformers import BertConfig

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import modelling
import beta_schedules
import sampling
import utils


def sample(
    num: int,
    dset_obj,
    model_path: str,
    config_json: Optional[str] = None,
    seed: int = 6489,
) -> List[torch.Tensor]:
    """
    Sample from the given model
    """
    assert hasattr(
        dset_obj, "sample_length"
    ), "Passed dataset object must have a sample_length attribute"
    # Load in the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query")
    model = modelling.BertForDiffusion.load_from_checkpoint(
        checkpoint_path=model_path, config=cfg
    )
    model.eval()
    model.to(device)

    # Reproduce the variance schedules bsaed on the config json
    if config_json is None:
        # Try to find a default config
        config_json = os.path.join(os.path.dirname(model_path, "training_args.json"))
        assert os.path.isfile(
            config_json
        ), f"Could not automatically find config at {config_json}"
    with open(config_json) as source:
        model_config = json.load(source)
    betas = beta_schedules.get_variance_schedule(
        model_config["variance_schedule"], model_config["timesteps"]
    )

    # Calculate posterior variance
    alphas = 1.0 - betas
    # corresponds to bar alpha, product up till t of the first t 1-B terms
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # Posterior variance, higher variance wih greater t
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # Sample
    # batch 128 ~ 9GB GPU memory, batch 512 ~ 38GB GPU memory
    torch.manual_seed(seed)
    samps = []
    for bs in utils.num_to_groups(num, 512):
        seq_lens = [dset_obj.sample_length() for _ in range(bs)]
        s = sampling.sample(
            model,
            seq_lens=seq_lens,
            seq_max_len=model.config.max_position_embeddings,
            betas=betas,
            posterior_variance=posterior_variance,
            timesteps=model_config["timesteps"],
            batch_size=bs,
            noise_modulo=[0, 2 * np.pi, 2 * np.pi, 2 * np.pi],
        )
        samps.extend(s)
    # samps = torch.vstack(samps)
    return samps


def main():
    import datasets

    cath_dset = datasets.CathConsecutiveAnglesDataset(split="train", toy=True)
    x = sample(
        10,
        cath_dset,
        "/home/t-kevinwu/projects/protein_diffusion/models_initial/1000_timesteps_linear_variance_schedule_64_batch_size_0.0001_lr_0.5_gradient_clip/lightning_logs/version_0/checkpoints/epoch=9-step=1990.ckpt",
        "/home/t-kevinwu/projects/protein_diffusion/models_initial/1000_timesteps_linear_variance_schedule_64_batch_size_0.0001_lr_0.5_gradient_clip/training_args.json",
    )
    for item in x:
        print(item.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
