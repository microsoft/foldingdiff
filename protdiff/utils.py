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


def tolerant_comparison_check(values, cmp: Literal[">=", "<="], v):
    """
    Compares values in a way that is tolerant of numerical precision
    >>> tolerant_comparison_check(np.array([1.1, 1.1]), ">=", 1.0)
    True
    >>> tolerant_comparison_check(np.array([1.1, 1.1]), "<=", 1.0)
    False
    >>> tolerant_comparison_check(np.array([1.1, 1.1]), "<=", 1.2)
    True
    >>> tolerant_comparison_check(np.array([1.1, 1.1]), ">=", 1.2)
    False
    >>> tolerant_comparison_check(-np.array([1.1, 1.1]), ">=", -1.2)
    True
    >>> tolerant_comparison_check(-np.array([1.1, 1.1]), "<=", -1.0)
    True
    >>> tolerant_comparison_check(-np.array([1.1, 1.1]), "<=", -1.2)
    False
    >>> tolerant_comparison_check(-np.array([1.1, 1.1]), ">=", -1.0)
    False
    >>> tolerant_comparison_check(-3.1415927410125732, ">=", -np.pi)
    True
    """
    if cmp == ">=":  # v is a lower bound
        minval = np.nanmin(values)
        diff = minval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True  # Passes
        return diff > 0
    elif cmp == "<=":
        maxval = np.nanmax(values)
        diff = maxval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True
        return diff < 0
    else:
        raise ValueError(f"Illegal comparator: {cmp}")


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
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
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
