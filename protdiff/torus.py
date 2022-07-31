"""
Functions for score based matching on a torus

See:
* https://github.com/gcorso/torsional-diffusion/blob/fcad6fb70da275ea7131be2aa5a3941e2c1129de/diffusion/torus.py#L33
* https://arxiv.org/pdf/2206.01729.pdf
"""
import logging
import numpy as np
import tqdm
import os

logging.basicConfig(level=logging.INFO)


def p(x, sigma, N: int = 10) -> np.ndarray:
    """
    Calculate the probabilities
    """
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N: int = 10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (
            (x + 2 * np.pi * i)
            / sigma ** 2
            * np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma ** 2)
        )
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

# For values of x and sigma, pre-compute the
x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi
logging.info(
    f"Computing scores for {x.shape} values of x and {sigma.shape} values of sigma"
)

if os.path.exists(".p.npy"):
    p_ = np.load(".p.npy")
    score_ = np.load(".score.npy")
else:
    p_ = p(x, sigma[:, None], N=100)
    np.save(".p.npy", p_)

    score_ = grad(x, sigma[:, None], N=100) / p_
    np.save(".score.npy", score_)

logging.info(f"p matrix of shape     {p_.shape}")
logging.info(f"Score matrix of shape {score_.shape}")


def score(x, sigma):
    # Mod 2pi on x
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    # Discretize x
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    # Discretize sigma
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten(),
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
