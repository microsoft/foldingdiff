"""
Utility functions for plotting
"""
import os, sys
from typing import Optional

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
assert os.path.isdir(PLOT_DIR)


def plot_val_dists_at_t(
    dset,
    t: int,
    share_axes: bool = True,
    zero_center_angles: bool = False,
    fname: Optional[str] = None,
):
    select_by_attn = lambda x: x["corrupted"][torch.where(x["attn_mask"])]

    retval = []
    for i in tqdm(range(len(dset))):
        vals = dset.__getitem__(i, use_t_val=t)
        assert vals["t"].item() == t, f"Unexpected values of t: {vals['t']} != {t}"
        retval.append(select_by_attn(vals))
    vals_flat = torch.vstack(retval).numpy()
    assert len(vals_flat.shape) == 2

    fig, axes = plt.subplots(
        2, 2, sharex=share_axes, sharey=share_axes, dpi=300, figsize=(9, 7)
    )
    for i, ax in enumerate(axes.flatten()):
        val_name = ["dist", "omega", "theta", "phi"][i]
        # Plot the values
        vals = vals_flat[:, i]
        sns.histplot(vals, ax=ax)
        if val_name != "dist":
            if zero_center_angles:
                ax.axvline(np.pi, color="tab:orange")
                ax.axvline(-np.pi, color="tab:orange")
            else:
                ax.axvline(0, color="tab:orange")
                ax.axvline(2 * np.pi, color="tab:orange")
        ax.set(title=f"Timestep {t} - {val_name}")
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_losses(log_fname: str, out_fname: Optional[str] = None, simple: bool = False):
    """
    Plot the validation loss values from a log file. Spuports multiple
    validation losses if present in log file.
    """
    fig, ax = plt.subplots(dpi=300)

    df = pd.read_csv(log_fname)
    for colname in df.columns:
        if "loss" not in colname:
            continue
        if simple and colname not in ["train_loss", "val_loss"]:
            continue
        vals = df.loc[:, ["epoch", colname]]
        vals.dropna(axis="index", how="any", inplace=True)
        ax.plot(vals["epoch"], vals[colname], label=colname, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")

    if out_fname is not None:
        fig.savefig(out_fname, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    plot_losses(sys.argv[1], out_fname=sys.argv[2], simple=True)
