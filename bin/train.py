"""
Training script
"""

import os, sys
import logging
from pathlib import Path
import multiprocessing
import argparse

from tqdm.auto import tqdm

import numpy as np

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


def train(
    batch_size: int = 128,
    lr: float = 1e-4,
    epochs: int = 3,
    device: torch.DeviceObjType = torch.device("cpu"),
):
    cath_dset = datasets.CathConsecutiveAnglesDataset(toy=False)
    noised_cath_dset = datasets.NoisedAnglesDataset(cath_dset)
    dataloader = DataLoader(
        dataset=noised_cath_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query",)
    model = modelling.BertForDiffusion(cfg)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in (pbar := tqdm(range(epochs))) :
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            known_noise = batch["known_noise"]
            predicted_noise = model(batch["corrupted"], batch["t"])

            loss = F.smooth_l1_loss(known_noise, predicted_noise)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                pbar.set_description(
                    f"Epoch {epoch} loss: {np.mean(epoch_losses[-50:]):.4f}"
                )


def main():
    train(device=torch.device("cuda"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
