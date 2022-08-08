"""
Loss functions!
"""

import numpy as np

import torch
from torch import nn

import utils


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
    input: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    circle_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Smooth radian L1 loss
    if the abs(delta) < beta --> 0.5 * delta^2 / beta
    else --> abs(delta) - 0.5 * beta

    See:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#smooth_l1_loss

    >>> radian_smooth_l1_loss(torch.tensor(0.1), 2 * torch.pi, beta=1.0)
    tensor(0.0050)
    >>> radian_smooth_l1_loss(torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1), beta=1.0)
    tensor(0.0200)
    >>> radian_smooth_l1_loss(torch.tensor(0.0), torch.tensor(3.14), beta=1.0)
    tensor(2.6400)
    >>> radian_smooth_l1_loss(torch.tensor(2.), torch.tensor(4.), beta=1.0)
    tensor(1.5000)
    >>> radian_smooth_l1_loss(torch.tensor(-0.1), torch.tensor(torch.pi + 2), beta=0.1) # 
    tensor(0.9916)
    >>> radian_smooth_l1_loss(torch.tensor(0.5), torch.tensor(-torch.pi), beta=0.1) # 3.14 - 0.5 = 2.64 - 0.1 / 2 = 2.59
    tensor(2.5916)
    >>> radian_smooth_l1_loss(torch.tensor(0.), torch.tensor(- 2 * torch.pi), beta=0.1)
    tensor(0.)
    >>> radian_smooth_l1_loss(torch.tensor(-17.0466), torch.tensor(-1.3888), beta=0.1)
    tensor(3.0414)
    """
    assert beta > 0
    d = target - input
    d = utils.modulo_with_wrapped_range(d, -torch.pi, torch.pi)

    abs_d = torch.abs(d)
    retval = torch.where(abs_d < beta, 0.5 * (d ** 2) / beta, abs_d - 0.5 * beta)
    assert torch.all(retval >= 0)
    retval = torch.mean(retval)

    # Regularize on "turns" around the circle
    if circle_penalty > 0:
        retval += circle_penalty * torch.mean(
            torch.div(torch.abs(input), torch.pi, rounding_mode="trunc")
        )

    return retval


def main():
    l = radian_smooth_l1_loss(torch.tensor(-17.0466), torch.tensor(-1.3888), beta=0.1)
    print(l)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
