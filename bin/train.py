"""
Training script
"""

import os, sys
import shutil
import logging
from pathlib import Path
import multiprocessing
import argparse

from tqdm.auto import tqdm

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import BertConfig

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))

import datasets
import modelling
from beta_schedules import SCHEDULES


# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


def plot_epoch_losses(loss_values, fname: str):
    """Plot the loss values and save to fname"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(np.arange(len(loss_values)), loss_values)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")
    fig.savefig(fname)


def train(
    results_dir: str = "./results",
    timesteps: int = 1000,
    variance_schedule: SCHEDULES = "linear",
    batch_size: int = 128,
    lr: float = 1e-4,
    epochs: int = 5,
    device: torch.DeviceObjType = torch.device("cpu"),
    multithread: bool = True,
):
    """Main training loop"""
    # Create results directory
    results_folder = Path(results_dir)
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(exist_ok=True)

    # Create dataset
    cath_dset = datasets.CathConsecutiveAnglesDataset(toy=False)
    noised_cath_dset = datasets.NoisedAnglesDataset(
        cath_dset,
        dset_key="angles",
        timesteps=timesteps,
        beta_schedule=variance_schedule,
    )
    dataloader = DataLoader(
        dataset=noised_cath_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() if multithread else 1,
    )

    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query",)
    model = modelling.BertForDiffusion(cfg)
    model.to(device)

    per_epoch_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in (pbar := tqdm(range(epochs))) :
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            known_noise = batch["known_noise"]
            predicted_noise = model(
                batch["corrupted"], batch["t"], attention_mask=batch["attn_mask"]
            )

            # COmpute loss on unmasked positions
            unmask_idx = torch.where(batch["attn_mask"])
            loss = F.smooth_l1_loss(
                known_noise[unmask_idx], predicted_noise[unmask_idx]
            )
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                pbar.set_description(
                    f"Epoch {epoch} loss: {np.mean(epoch_losses[-50:]):.4f}"
                )
        per_epoch_losses.append(np.mean(epoch_losses))

    plot_epoch_losses(per_epoch_losses, results_folder / "losses.pdf")


def main():
    train(epochs=5, device=torch.device("cuda"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
