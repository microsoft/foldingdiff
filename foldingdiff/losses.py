"""
Loss functions!
"""
from typing import Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from foldingdiff import utils


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
    >>> radian_smooth_l1_loss(torch.tensor(-17.0466), torch.tensor(-1.3888), beta=0.1)
    tensor(3.0414)
    """
    assert (
        target.shape == input.shape
    ), f"Mismatched shapes: {input.shape} != {target.shape}"
    assert beta > 0
    d = target - input
    d = utils.modulo_with_wrapped_range(d, -torch.pi, torch.pi)

    abs_d = torch.abs(d)
    retval = torch.where(abs_d < beta, 0.5 * (d**2) / beta, abs_d - 0.5 * beta)
    assert torch.all(retval >= 0), f"Got negative loss terms: {torch.min(retval)}"
    retval = torch.mean(retval)

    # Regularize on "turns" around the circle
    if circle_penalty > 0:
        retval += circle_penalty * torch.mean(
            torch.div(torch.abs(input), torch.pi, rounding_mode="trunc")
        )

    return retval


def _get_pairwise_dist_batch(
    values: torch.Tensor, lengths: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the pairwise distance matrix for the given values
    Returns two tensors of shape (batch, M) where M is the number of pairwise
    distances, padded. First tensor is actual distances, second tensor is a mask.
    Mask is formatted such that valid values are 1 and invalid values are 0.
    """
    assert values.ndim == 3, f"Expected 3D tensor of (batch, N, 3), got {values.ndim}D"
    assert (
        values.shape[-1] == 3
    ), f"Expected 3D tensor of (batch, N, 3), got {values.shape}"
    assert lengths.ndim == 1
    assert values.shape[0] == lengths.shape[0]

    # Calculate the pairwise distances
    dev = values.device
    values = [F.pdist(values[i, :l]) for i, l in enumerate(lengths)]

    # Pad
    lengths = [len(v) for v in values]
    max_len = max(lengths)
    mask = torch.zeros((len(values), max_len))
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    retval = torch.zeros((len(values), max_len)).to(dev)
    for i, v in enumerate(values):
        retval[i, : len(v)] = v
    assert retval.shape == mask.shape

    return retval, mask


def pairwise_dist_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
):
    """
    Calculates the pairwise distance matrix for both the input and the target,
    which are given in 3D cartesian coordinates of the shape (batch, N, 3) and
    calculates a loss based on the difference in the pairwise distances. Padding
    is handled using the lengths tensor, which is a 1D tensor of shape (batch,)
    and contains the number of valid values in each batch. Pairwise distances are
    calculated betwen all given coordinates.

    Note that since we are comparing pairwise distances, this loss function itself
    is rotation and shift invariant.
    """
    if lengths is None:
        lengths = torch.IntTensor(
            [
                torch.all(~torch.isnan(input[i]), dim=1).sum()
                for i in range(input.shape[0])
            ]
        )
    assert lengths.shape[0] == input.shape[0]
    # Calculate pairwise distances; these return (batch, M) tensors
    # of the distances and a mask (where 0 indicates a padding value)
    input_dists, input_mask = _get_pairwise_dist_batch(input, lengths)
    target_dists, target_mask = _get_pairwise_dist_batch(target, lengths)
    assert torch.allclose(input_mask, target_mask)

    batch_indices, _seq_indices = torch.where(input_mask)

    # Get the loss
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
    loss = F.mse_loss(
        input_dists[torch.where(input_mask)],
        target_dists[torch.where(target_mask)],
        reduction="none",
    )
    # Scale by weights optionally
    if weights is not None:
        if weights.ndim > 1:
            assert weights.shape[0] == input.shape[0]
            # loss shape (batch, M), weights after indexing is (M,) --> (1, M)
            loss *= weights[batch_indices].squeeze()
        else:
            loss *= weights
    return torch.mean(loss)


def main():
    lengths = torch.randint(2, 5, size=(16,)) * 3
    x = torch.randn(16, 12, 3)
    y = torch.randn(16, 12, 3)

    l = pairwise_dist_loss(x, y, lengths)
    print(l)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
