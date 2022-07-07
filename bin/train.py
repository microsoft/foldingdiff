"""
Training script
"""

import os, sys
import shutil
import logging
from pathlib import Path
import multiprocessing
import argparse
from typing import *

from tqdm.auto import tqdm

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch import optim
from torch.utils.data import Dataset
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


def get_train_valid_test_sets(
    timesteps: int, variance_schedule: SCHEDULES
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the dataset objects to use for train/valid/test

    Note, these need to be wrapped in data loaders later
    """
    clean_dsets = [
        datasets.CathConsecutiveAnglesDataset(split=s)
        for s in ["train", "validation", "test"]
    ]
    noised_dsets = [
        datasets.NoisedAnglesDataset(
            ds, dset_key="angles", timesteps=timesteps, beta_schedule=variance_schedule,
        )
        for ds in clean_dsets
    ]
    return tuple(noised_dsets)


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

    # Get datasets and wrap them in dataloaders
    dsets = get_train_valid_test_sets(
        timesteps=timesteps, variance_schedule=variance_schedule
    )
    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count() if multithread else 1,
        )
        for ds in dsets
    ]

    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query",)
    model = modelling.BertForDiffusion(cfg)
    model.to(device)

    per_epoch_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in (pbar := tqdm(range(epochs))) :
        epoch_losses = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            # for k, v in batch.items():
            #     print(k, v.shape)
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

        # Evaluate on validation set
        with torch.no_grad():
            val_losses = []
            for batch in valid_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                known_noise = batch["known_noise"]
                # for k, v in batch.items():
                #     print(k, v.shape())
                predicted_noise = model(
                    batch["corrupted"], batch["t"], attention_mask=batch["attn_mask"]
                )
                unmask_idx = torch.where(batch["attn_mask"])
                loss = F.smooth_l1_loss(
                    known_noise[unmask_idx], predicted_noise[unmask_idx]
                )
                val_losses.append(loss.item())
            logging.info(f"Epoch {epoch} validation loss: {np.mean(val_losses)}")

    plot_epoch_losses(per_epoch_losses, results_folder / "losses.pdf")


def main():
    train(epochs=5, device=torch.device("cuda"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
