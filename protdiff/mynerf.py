"""
NERF!

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
"""

import numpy as np


def place_dihedral(a, b, c, bond_angle, bond_length, torsion_angle):
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    """
    ab = b - a
    bc = c - b
    bcn = bc / np.linalg.norm(bc)
    # numpy is row major
    d = np.array(
        [
            -bond_length * np.cos(bond_angle),
            bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
            bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
        ]
    )
    n = np.cross(ab, bcn)
    n /= np.linalg.norm(n)
    nbc = np.cross(n, bcn)
    m = np.array(
        [[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]], [bcn[2], nbc[2], n[2]]]
    )
    d = m.dot(d)
    d = d + c
    return d


if __name__ == "__main__":
    rng = np.random.default_rng(seed=1)
    a, b, c, d = rng.uniform(low=-5, high=5, size=(4, 3))
    print(a, b, c, d)
