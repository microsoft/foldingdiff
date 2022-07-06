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

from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import BertConfig

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))

import datasets
import modelling


def main():
    cath_dset = datasets.CathConsecutiveAnglesDataset(toy=True)
    noised_cath_dset = datasets.NoisedAnglesDataset(cath_dset)
    dataloader = DataLoader(
        dataset=noised_cath_dset,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    cfg = BertConfig(hidden_size=144, position_embedding_type="relative_key_query",)
    model = modelling.BertForDiffusion(cfg)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in (pbar := tqdm(range(2))) :
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # batch = {k: v.to(device) for k, v in batch.items()}
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
