"""
Test for code related to converting angles to coordinates
"""
import os, sys
import unittest

import numpy as np

from sequence_models import pdb_utils

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import angles_to_coords


class TestDihedralReversalEasy(unittest.TestCase):
    def test_easy(self):
        """
        Easy test where we simply have a right angle
        """
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        d = np.array([[-1.0, 1.0, 0.0]])
        dh = pdb_utils.get_dihedrals(a, b, c, d)

        rev = angles_to_coords.reverse_dihedral(a, b, c, dh)
        self.assertTrue(np.allclose((d - c).squeeze(), rev))

    def test_scale_invariant_bc(self):
        """
        Test that a longer distance between b/c does not influence calculation
        """
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 10.0, 0.0]])
        d = np.array([[-1.0, 10.0, 0.0]])
        dh = pdb_utils.get_dihedrals(a, b, c, d)

        rev = angles_to_coords.reverse_dihedral(a, b, c, dh)
        self.assertTrue(np.allclose((d - c).squeeze(), rev))

    def test_scale_invariant_ab(self):
        """
        Test that a longer distance between a/b does not influence calculation
        """
        a = np.array([[10.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        d = np.array([[-1.0, 1.0, 0.0]])
        dh = pdb_utils.get_dihedrals(a, b, c, d)

        rev = angles_to_coords.reverse_dihedral(a, b, c, dh)
        self.assertTrue(np.allclose((d - c).squeeze(), rev))

    def test_scale_invariant_cd(self):
        """
        Test that a longer distance between c/d does not influence calculation
        """
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        d = np.array([[-10.0, 1.0, 0.0]])
        dh = pdb_utils.get_dihedrals(a, b, c, d)

        expected = d - c
        expected /= np.linalg.norm(expected)
        rev = angles_to_coords.reverse_dihedral(a, b, c, dh)
        self.assertTrue(
            np.allclose(expected, rev), msg="{} != {}".format(expected, rev)
        )


class TestDihedralReversalRandom(unittest.TestCase):
    def test_random(self):
        rng = np.random.default_rng(seed=6489)
        for _ in range(10):
            a, b, c, d = rng.random((4, 1, 3))
            dh = pdb_utils.get_dihedrals(a, b, c, d)
            expected = d - c
            expected /= np.linalg.norm(expected)
            rev = angles_to_coords.reverse_dihedral(a, b, c, dh)
            self.assertTrue(
                np.allclose(expected, rev), msg="{} != {}".format(expected, rev)
            )


if __name__ == "__main__":
    unittest.main()
