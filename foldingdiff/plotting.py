"""
Utility functions for plotting
"""
import os, sys
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import torch
from torch.utils.data import Dataset

PLOT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))
if not PLOT_DIR.is_dir():
    os.makedirs(PLOT_DIR)


def plot_joint_kde(
    x_values, y_values, show_axes: bool = True, fname: Optional[str] = None, **kwargs
):
    """
    Plot a density scatter plot (KDE) of the values. kwargs are passed to
    ax.set()

    Useful for plotting Ramachandran plot - x = phi, y = psi
    https://proteopedia.org/wiki/index.php/Ramachandran_Plots
    """
    fig, ax = plt.subplots(dpi=300)
    sns.kdeplot(x=x_values, y=y_values, levels=100, fill=True, norm=LogNorm(), ax=ax)
    if show_axes:
        ax.axvline(0, color="grey", alpha=0.5)
        ax.axhline(0, color="grey", alpha=0.5)
    ax.set(**kwargs)
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_val_dists_at_t(
    t: int,
    dset: Dataset,
    share_axes: bool = True,
    zero_center_angles: bool = False,
    fname: Optional[str] = None,
):
    select_by_attn = lambda x: x["corrupted"][torch.where(x["attn_mask"])]

    retval = []
    for i in range(len(dset)):
        vals = dset.__getitem__(i, use_t_val=t)
        assert vals["t"].item() == t, f"Unexpected values of t: {vals['t']} != {t}"
        retval.append(select_by_attn(vals))
    vals_flat = torch.vstack(retval).numpy()
    assert vals_flat.ndim == 2

    ft_names = dset.feature_names["angles"]
    n_fts = len(ft_names)
    assert vals_flat.shape[1] == n_fts

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_fts,
        sharex=share_axes,
        sharey=share_axes,
        dpi=300,
        figsize=(2.6 * n_fts, 2.5),
    )
    for i, (ax, ft_name) in enumerate(zip(axes, ft_names)):
        # Plot the values
        vals = vals_flat[:, i]
        sns.histplot(vals, ax=ax)
        if "dist" not in ft_name:
            if zero_center_angles:
                ax.axvline(np.pi, color="tab:orange")
                ax.axvline(-np.pi, color="tab:orange")
            else:
                ax.axvline(0, color="tab:orange")
                ax.axvline(2 * np.pi, color="tab:orange")
        ax.set(title=f"Timestep {t} - {ft_name}")
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_losses(
    log_fname: str,
    out_fname: Optional[str] = None,
    simple: bool = False,
    pattern: Optional[str] = None,
):
    """
    Plot the validation loss values from a log file. Spuports multiple
    validation losses if present in log file. Plots per epoch, and if multiple
    values are record for an epoch, plot the median.
    """

    def keyfunc(x: str) -> tuple:
        """
        Validation first, then train
        """
        ordering = ["test", "val", "train"]
        if "_" in x:
            x_split, x_val = x.split("_", maxsplit=1)
            x_retval = tuple([ordering.index(x_split), x_val])
        else:
            x_retval = (len(ordering), x)
        assert len(x_retval) == 2
        return x_retval

    if simple:
        assert pattern is None
        pattern = re.compile(r"_loss$")
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    fig, ax = plt.subplots(dpi=300)

    df = pd.read_csv(log_fname)
    cols = df.columns.to_list()
    cols = sorted(cols, key=keyfunc)
    for colname in df.columns:
        if "loss" not in colname:
            continue
        if pattern is not None:
            if not pattern.search(colname):
                continue
        vals = df.loc[:, ["epoch", colname]]
        vals.dropna(axis="index", how="any", inplace=True)
        sns.lineplot(x="epoch", y=colname, data=vals, ax=ax, label=colname, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")

    if out_fname is not None:
        fig.savefig(out_fname, bbox_inches="tight")
    return fig


def plot_consecutive_heatmap(
    vals: Union[Sequence[float], Sequence[Sequence[float]]],
    fname: Optional[str] = None,
    logstretch_vmax: float = 2e3,
    **kwargs,
):
    """
    Plot a heatmap of consecutive values.
    """
    consecutive_pairs = []

    get_pairs = lambda x: np.array(list(zip(x[:-1], x[1:])))
    # Require these more complex checks because vals may not be of the same
    # size and therefore may not be stackable
    if isinstance(vals[0], (float, int)):
        # 1-dimensional
        consecutive_pairs = get_pairs(vals)
    else:
        # 2-dimensional
        consecutive_pairs = np.vstack([get_pairs(vec) for vec in vals])
    assert consecutive_pairs.ndim == 2
    assert consecutive_pairs.shape[1] == 2

    norm = ImageNormalize(vmin=0.0, vmax=logstretch_vmax, stretch=LogStretch())

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(
        consecutive_pairs[:, 0], consecutive_pairs[:, 1], norm=norm
    )
    fig.colorbar(density, label="Points per pixel")

    ax.set(**kwargs)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    plot_losses(sys.argv[1], out_fname=sys.argv[2], simple=True)
