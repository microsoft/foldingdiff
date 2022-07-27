"""
Loss functions!
"""

import numpy as np

import torch
from torch import nn


def radian_l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss between input and target
    >>> radian_l1_loss(torch.tensor(0.1), 2 * torch.pi)
    tensor(0.1000)
    >>> radian_l1_loss(torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1))
    tensor(0.2000)
    """
    # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
    target = target % (2 * torch.pi)
    input = input % (2 * torch.pi)
    d = target - input
    d = (d + torch.pi) % (2 * torch.pi) - torch.pi
    retval = torch.abs(d)
    return torch.mean(retval)


def radian_smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """
    Smooth radian L1 loss
    if the abs(delta) < beta --> 0.5 * delta^2 / beta
    else --> abs(delta) - 0.5 * beta

    See:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#smooth_l1_loss

    >>> radian_smooth_l1_loss(torch.tensor(0.1), 2 * torch.pi)
    tensor(0.0050)
    >>> radian_smooth_l1_loss(torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1))
    tensor(0.0200)
    >>> radian_smooth_l1_loss(torch.tensor(0.0), torch.tensor(3.14))
    tensor(2.6400)
    >>> radian_smooth_l1_loss(torch.tensor(2.), torch.tensor(4.))
    tensor(1.5000)
    """
    target = target % (2 * torch.pi)
    input = input % (2 * torch.pi)
    d = target - input
    d = (d + torch.pi) % (2 * torch.pi) - torch.pi
    d = torch.abs(d)

    retval = torch.where(d < beta, 0.5 * d ** 2 / beta, abs(d) - 0.5 * beta)
    return torch.mean(retval)


def main():
    l = radian_l1_loss(torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1))
    print(l)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
