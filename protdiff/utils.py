"""
Misc shared utility functions
"""
import logging
from typing import *

import numpy as np
import torch


def extract(a, t, x_shape):
    """
    Return the t-th item in a for each item in t
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def num_to_groups(num: int, divisor: int) -> List[int]:
    """
    Generates a list of ints of value at most divisor that sums to

    >>> num_to_groups(18, 16)
    [16, 2]
    >>> num_to_groups(33, 8)
    [8, 8, 8, 8, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    assert sum(arr) == num
    return arr


def broadcast_mod(x: torch.Tensor, m: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Perform modulo on x % m while broadcasting. values in m that are 0 are ignored.
    >>> broadcast_mod(torch.arange(24).reshape(2, 3, 4), torch.tensor([5, 7, 9, 11]))
    tensor([[[0, 1, 2, 3],
             [4, 5, 6, 7],
             [3, 2, 1, 0]],
    <BLANKLINE>
            [[2, 6, 5, 4],
             [1, 3, 0, 8],
             [0, 0, 4, 1]]])
    >>> broadcast_mod(torch.arange(24).reshape(2, 3, 4), torch.tensor([0, 7, 9, 11]))
    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  2,  1,  0]],
    <BLANKLINE>
            [[12,  6,  5,  4],
             [16,  3,  0,  8],
             [20,  0,  4,  1]]])
    """
    if isinstance(m, float):
        assert m != 0
        return torch.remainder(x, m)
    m = m.to(x.device)
    # m is a tensor so we need to broadcast
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    m_clipped = m.clamp(min=1)
    moddded = torch.remainder(x, m_clipped.expand(*x.shape))
    retval = torch.where(m == 0, x, moddded)
    return retval


def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    if isinstance(retval, torch.Tensor):
        notnan_idx = ~torch.isnan(retval)
        assert torch.all(retval[notnan_idx] >= range_min)
        assert torch.all(retval[notnan_idx] < range_max)
    else:
        assert np.all(
            np.nanmin(retval) >= range_min
        ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
        assert np.all(
            np.nanmax(retval) <= range_max
        ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval


def update_dict_nonnull(d: Dict[str, Any], vals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a dictionary with values from another dictionary.
    >>> update_dict_nonnull({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}
    """
    for k, v in vals.items():
        if k in d:
            if d[k] != v and v is not None:
                logging.info(f"Replacing key {k} original value {d[k]} with {v}")
                d[k] = v
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    import doctest

    doctest.testmod()
