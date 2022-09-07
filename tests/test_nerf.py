"""
Unit tests for NERF conversion of internal coordinates to cartesian coordinates
"""

import os, sys
import unittest

import numpy as np
from sequence_models import pdb_utils
from biotite.structure import dihedral

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import mynerf as nerf


class TestDihedralPlacement(unittest.TestCase):
    """
    Test the dihedral placement
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=6489)

    def test_simple(self):
        """Simple test about origin"""
        a = np.array([1, 0, 0])
        b = np.array([0, 0, 0])
        c = np.array([0, 1, 0])
        d = np.array([0, 1, 1])
        calc_d = nerf.place_dihedral(a, b, c, np.pi / 2, 1.0, -np.pi / 2)
        self.assertTrue(np.allclose(d, calc_d), f"Mismatched: {d} != {calc_d}")

    def test_randomized(self):
        """Simple test using randomized values"""
        for _ in range(100):
            a, b, c, d = self.rng.uniform(low=-5, high=5, size=(4, 3))
            calc_d = nerf.place_dihedral(
                a,
                b,
                c,
                angle_between(d - c, b - c),
                dist_between(c, d),
                dihedral(a, b, c, d),
            )
            self.assertTrue(np.allclose(d, calc_d), f"Mismatched: {d} != {calc_d}")


def angle_between(v1, v2) -> float:
    """Gets the angle between u and v"""
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dist_between(a, b):
    """Distance between a and b"""
    d = a - b
    return np.linalg.norm(d, 2)


if __name__ == "__main__":
    unittest.main()
