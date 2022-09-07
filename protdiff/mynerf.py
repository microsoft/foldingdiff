"""
NERF!

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
"""

import numpy as np


def place_dihedral(a: np.ndarray, b:np.ndarray, c: np.ndarray, bond_angle:float, bond_length:float, torsion_angle:float) -> np.ndarray:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.ndim == b.ndim == c.ndim == 1
    unit_vec = lambda x: x / np.linalg.norm(x)
    ab = b - a
    bc = unit_vec(c - b)
    d = np.array(
        [
            -bond_length * np.cos(bond_angle),
            bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
            bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
        ]
    )
    n = unit_vec(np.cross(ab, bc))
    nbc = np.cross(n, bc)
    m = np.stack([bc, nbc, n]).T
    d = m.dot(d)
    return d + c


if __name__ == "__main__":
    rng = np.random.default_rng(seed=1)
    a, b, c, d = rng.uniform(low=-5, high=5, size=(4, 3))
    print(a, b, c, d)
