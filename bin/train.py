"""
Training script
"""

import os, sys
import shutil
import json
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

import pytorch_lightning as pl

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
    multithread: bool = True,
):
    """Main training loop"""
    # Record the args given to the function before we create more vars
    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
    func_args = locals()

    # Create results directory
    results_folder = Path(results_dir)
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(exist_ok=True)
    with open(results_folder / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args, sink, indent=4)

    # Get datasets and wrap them in dataloaders
    dsets = get_train_valid_test_sets(
        timesteps=timesteps, variance_schedule=variance_schedule
    )
    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=i == 0,  # Shuffle only train loader
            num_workers=multiprocessing.cpu_count() if multithread else 1,
        )
        for i, ds in enumerate(dsets)
    ]

    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query",)
    model = modelling.BertForDiffusion(cfg, lr=lr)

    trainer = pl.Trainer(
        default_root_dir=results_folder,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


def main():
    train(epochs=5, device="cuda")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
