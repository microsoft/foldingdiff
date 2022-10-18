"""
Some custom metrics
"""
import functools
import multiprocessing
import logging

import numpy as np
from scipy import stats

import torch
from torch.utils.data import Dataset


def kl_from_empirical(
    u: np.ndarray, v: np.ndarray, nbins: int = 100, pseudocount: bool = False
) -> float:
    """
    Compute the KL divergence between two empirical distributions u and v.

    Discretizes the u and v distributions using nbins bins
    """
    min_val = min(np.min(u), np.min(v))
    max_val = max(np.max(u), np.max(v))
    logging.debug(f"Creating {nbins} bins between {min_val} - {max_val}")

    bins = np.linspace(min_val, max_val, nbins + 1)
    if pseudocount:
        u = np.concatenate((u, bins))
        v = np.concatenate((v, bins))
    u_hist, _u_bin_edges = np.histogram(u, bins=bins, density=True)
    v_hist, _v_bin_edges = np.histogram(v, bins=bins, density=True)

    # Compute KL divergence
    # https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
    kl = stats.entropy(u_hist, v_hist)
    return kl


def _kl_helper(t: int, dset: Dataset) -> np.ndarray:
    """
    Compute the KL divergence for each feature at timestep t
    Returns an array of size (n_features,) corresponding to KL divergence
    dset should be NoisedAnglesDataset
    """
    assert hasattr(dset, "feature_names")
    assert hasattr(dset, "sample_noise")
    select_by_attn = lambda x: x["corrupted"][torch.where(x["attn_mask"])]
    values = []
    for i in range(len(dset)):
        vals = dset.__getitem__(i, use_t_val=t)
        values.append(select_by_attn(vals).numpy())  # Non-masked positions

    values = np.vstack(values)
    assert values.ndim == 2
    assert values.shape[1] == len(dset.feature_names["angles"])

    # compute KL at each timestep
    noise = dset.sample_noise(torch.from_numpy(values)).numpy()
    kl_values = [
        kl_from_empirical(values[:, i], noise[:, i]) for i in range(values.shape[1])
    ]
    return np.array(kl_values)


def kl_from_dset(dset: Dataset, single_thread: bool = False) -> np.ndarray:
    """
    For each timestep in the dataset, compute the KL divergence across each feature
    """
    assert hasattr(dset, "timesteps")
    pfunc = functools.partial(_kl_helper, dset=dset)
    if single_thread:
        kl_values = [pfunc(t) for t in range(dset.timesteps)]
    else:
        logging.info(
            f"Computing KL divergence for {dset.timesteps} timesteps using {multiprocessing.cpu_count()} workers"
        )
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        kl_values = pool.map(pfunc, range(dset.timesteps))
        pool.close()
        pool.join()
    return np.array(kl_values)


def wrapped_mean(x: np.ndarray, axis=None) -> float:
    """
    Wrap the mean function about [-pi, pi]
    """
    # https://rosettacode.org/wiki/Averages/Mean_angle
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    retval = np.arctan2(np.nanmean(sin_x, axis=axis), np.nanmean(cos_x, axis=axis))
    return retval
