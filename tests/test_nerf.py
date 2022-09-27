"""
Unit tests for NERF conversion of internal coordinates to cartesian coordinates
"""

import os
import tempfile
import unittest

import numpy as np
from biotite.structure import dihedral

from foldingdiff import nerf
from foldingdiff import angles_and_coords as ac
from foldingdiff import tmalign


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


class TestBackboneReconstruction(unittest.TestCase):
    """
    Test that we can successfully reconstruct the backbone of a simple protein
    """

    def setUp(self) -> None:
        self.pdb_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/1CRN.pdb"
        )
        assert os.path.isfile(self.pdb_file)

        self.exhaustive_angles = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
        self.exhaustive_dists = ["0C:1N", "N:CA", "CA:C"]

        self.minimal_angles = ["phi", "psi", "omega"]
        self.minimal_dists = []

    def test_full_reconstruction(self):
        """Test that we can get the same structure back"""
        angles = ac.canonical_distances_and_dihedrals(
            self.pdb_file,
            distances=self.exhaustive_dists,
            angles=self.exhaustive_angles,
        )
        with tempfile.TemporaryDirectory() as dirname:
            out_fname = os.path.join(dirname, "temp.pdb")
            ac.create_new_chain_nerf(
                out_fname,
                angles,
                angles_to_set=self.exhaustive_angles,
                dists_to_set=self.exhaustive_dists,
                center_coords=False,
            )
            score = tmalign.run_tmalign(self.pdb_file, out_fname)
        self.assertAlmostEqual(1.0, score)
    
    def test_full_reconstruction_with_centering(self):
        """Test that we can get the same structure back with centering"""
        angles = ac.canonical_distances_and_dihedrals(
            self.pdb_file,
            distances=self.exhaustive_dists,
            angles=self.exhaustive_angles,
        )
        with tempfile.TemporaryDirectory() as dirname:
            out_fname = os.path.join(dirname, "temp.pdb")
            ac.create_new_chain_nerf(
                out_fname,
                angles,
                angles_to_set=self.exhaustive_angles,
                dists_to_set=self.exhaustive_dists,
                center_coords=True,
            )
            score = tmalign.run_tmalign(self.pdb_file, out_fname)
        self.assertAlmostEqual(1.0, score)

    def test_minimal_reconstruction(self):
        """Test that we can get a close enough structure back with fewer angles"""
        angles = ac.canonical_distances_and_dihedrals(
            self.pdb_file,
            distances=self.minimal_dists,
            angles=self.minimal_angles,
        )
        with tempfile.TemporaryDirectory() as dirname:
            out_fname = os.path.join(dirname, "temp.pdb")
            ac.create_new_chain_nerf(
                out_fname,
                angles,
                angles_to_set=self.minimal_angles,
                dists_to_set=self.minimal_dists,
            )
            score = tmalign.run_tmalign(self.pdb_file, out_fname)
        self.assertGreater(score, 0.5)


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
