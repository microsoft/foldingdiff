"""
Some custom metrics
"""
import logging

import numpy as np
from scipy import stats


def kl_from_empirical(u: np.ndarray, v: np.ndarray, nbins: int = 100) -> float:
    """
    Compute the KL divergence between two empirical distributions u and v.

    Discretizes the u and v distributions using nbins bins
    """
    min_val = min(np.min(u), np.min(v))
    max_val = max(np.max(u), np.max(v))
    logging.debug(f"Creating bins between {min_val} - {max_val}")

    bins = np.linspace(min_val, max_val, nbins + 1)
    u_hist, _u_bin_edges = np.histogram(u, bins=bins, density=True)
    v_hist, _v_bin_edges = np.histogram(v, bins=bins, density=True)

    # Compute KL divergence
    # https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
    kl = stats.entropy(u_hist, v_hist)
    return kl
