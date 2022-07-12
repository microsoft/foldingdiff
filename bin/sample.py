"""
Script to sample from a trained diffusion model
"""
import os, sys
import json
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F

from transformers import BertConfig

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import modelling
import beta_schedules
import sampling


def sample(model_path: str, config_json: Optional[str] = None):
    """
    Sample from the given model
    """
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
        config_json = os.path.join(os.path.dirname(model_path, "config.json"))
        assert os.path.isfile(
            config_json
        ), f"Could not automiatcally find config at {config_json}"
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
    sampling.sample(
        model,
        seq_len=512,
        betas=betas,
        posterior_variance=posterior_variance,
        timesteps=model_config["timesteps"],
    )


def main():
    sample(
        "/home/t-kevinwu/projects/protein_diffusion/models/1000_timesteps_cosine_variance_schedule_128_batch_size_0.0001_lr_0.0_gradient_clip/lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt",
        "/home/t-kevinwu/projects/protein_diffusion/models/1000_timesteps_cosine_variance_schedule_128_batch_size_0.0001_lr_0.0_gradient_clip/training_args.json",
    )


if __name__ == "__main__":
    main()
