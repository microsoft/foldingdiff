"""
Unit tests for NERF conversion of internal coordinates to cartesian coordinates
"""

import os
import tempfile
import unittest

import numpy as np
import torch
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

        self.pdb_file_2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/7PFL.pdb"
        )
        assert os.path.isfile(self.pdb_file_2)

        self.exhaustive_angles = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
        self.exhaustive_dists = ["0C:1N", "N:CA", "CA:C"]

        self.minimal_angles = ["phi", "psi", "omega"]
        self.minimal_dists = []

    def test_full_reconstruction(self):
        """Test that we can get the same structure back"""
        for pdb_file in [self.pdb_file, self.pdb_file_2]:
            angles = ac.canonical_distances_and_dihedrals(
                pdb_file,
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
                score = tmalign.run_tmalign(pdb_file, out_fname)
            self.assertAlmostEqual(1.0, score)

    def test_full_reconstruction_with_centering(self):
        """Test that we can get the same structure back with centering"""
        for pdb_file in [self.pdb_file, self.pdb_file_2]:
            angles = ac.canonical_distances_and_dihedrals(
                pdb_file,
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
                score = tmalign.run_tmalign(pdb_file, out_fname)
            self.assertAlmostEqual(1.0, score)

    def test_minimal_reconstruction(self):
        """Test that we can get a close enough structure back with fewer angles"""
        for pdb_file in [self.pdb_file, self.pdb_file_2]:
            angles = ac.canonical_distances_and_dihedrals(
                pdb_file,
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
                score = tmalign.run_tmalign(pdb_file, out_fname)
            self.assertGreater(score, 0.5)

    def test_batch_reconstruction(self):
        """Test that we can process a batch of inputs of equal length"""
        for pdb_file in [self.pdb_file, self.pdb_file_2]:
            angles = ac.canonical_distances_and_dihedrals(
                pdb_file,
                distances=self.exhaustive_dists,
                angles=self.exhaustive_angles,
            )

            bs = 64

            phi = torch.from_numpy(angles["phi"].values).unsqueeze(0).repeat(bs, 1)
            psi = torch.from_numpy(angles["psi"].values).unsqueeze(0).repeat(bs, 1)
            omega = torch.from_numpy(angles["omega"].values).unsqueeze(0).repeat(bs, 1)
            tau = torch.from_numpy(angles["tau"].values).unsqueeze(0).repeat(bs, 1)
            ca_c_1n = (
                torch.from_numpy(angles["CA:C:1N"].values).unsqueeze(0).repeat(bs, 1)
            )
            c_1n_1ca = (
                torch.from_numpy(angles["C:1N:1CA"].values).unsqueeze(0).repeat(bs, 1)
            )
            assert phi.shape == (bs, len(angles))

            built = (
                nerf.nerf_build_batch(
                    phi,
                    psi,
                    omega,
                    tau,
                    ca_c_1n,
                    c_1n_1ca,
                )
                .detach()
                .numpy()
            )
            self.assertTrue(built.shape[0] == bs)

            for coords in built:
                with tempfile.TemporaryDirectory() as dirname:
                    out_fname = os.path.join(dirname, "temp.pdb")
                    ac.write_coords_to_pdb(
                        coords,
                        out_fname,
                    )
                    score = tmalign.run_tmalign(pdb_file, out_fname)
                    self.assertGreater(score, 0.95)

    def test_batch_reconstruction_diff_len(self):
        """
        Test that we can process a batch of inputs of different lengths
        padded by nan values
        """
        phi, psi, omega, tau, ca_c_1n, c_1n_1ca = [], [], [], [], [], []
        pdb_files = [self.pdb_file, self.pdb_file_2]
        for pdb_file in pdb_files:
            angles = ac.canonical_distances_and_dihedrals(
                pdb_file,
                distances=self.exhaustive_dists,
                angles=self.exhaustive_angles,
            )

            phi.append(torch.from_numpy(angles["phi"].values))
            psi.append(torch.from_numpy(angles["psi"].values))
            omega.append(torch.from_numpy(angles["omega"].values))
            tau.append(torch.from_numpy(angles["tau"].values))
            ca_c_1n.append(torch.from_numpy(angles["CA:C:1N"].values))
            c_1n_1ca.append(torch.from_numpy(angles["C:1N:1CA"].values))

        l = max([len(p) for p in phi])
        for i in range(len(phi)):
            phi[i] = torch.cat([phi[i], torch.zeros(l - len(phi[i])) * np.nan])
            psi[i] = torch.cat([psi[i], torch.zeros(l - len(psi[i])) * np.nan])
            omega[i] = torch.cat([omega[i], torch.zeros(l - len(omega[i])) * np.nan])
            tau[i] = torch.cat([tau[i], torch.zeros(l - len(tau[i])) * np.nan])
            ca_c_1n[i] = torch.cat(
                [ca_c_1n[i], torch.zeros(l - len(ca_c_1n[i])) * np.nan]
            )
            c_1n_1ca[i] = torch.cat(
                [c_1n_1ca[i], torch.zeros(l - len(c_1n_1ca[i])) * np.nan]
            )

        phi = torch.stack(phi)
        psi = torch.stack(psi)
        omega = torch.stack(omega)
        tau = torch.stack(tau)
        ca_c_1n = torch.stack(ca_c_1n)
        c_1n_1ca = torch.stack(c_1n_1ca)

        built = (
            nerf.nerf_build_batch(
                phi,
                psi,
                omega,
                tau,
                ca_c_1n,
                c_1n_1ca,
            )
            .detach()
            .numpy()
        )
        self.assertTrue(built.shape[0] == 2)

        for i, (coords, pdb_file) in enumerate(zip(built, pdb_files)):
            with tempfile.TemporaryDirectory() as dirname:
                out_fname = os.path.join(dirname, "temp.pdb")
                ac.write_coords_to_pdb(
                    coords[np.all(~np.isnan(coords), axis=1)],
                    out_fname,
                )
                score = tmalign.run_tmalign(pdb_file, out_fname)
                self.assertGreater(score, 0.95)


class TestPytorchBackend(unittest.TestCase):
    """
    Test that PyTorch backend should work
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=6489)

    def test_simple(self):
        """Simple test about origin"""
        a = torch.tensor([1, 0, 0], dtype=torch.float64)
        b = torch.tensor([0, 0, 0], dtype=torch.float64)
        c = torch.tensor([0, 1, 0], dtype=torch.float64)
        d = torch.tensor([0, 1, 1], dtype=torch.float64)
        calc_d = nerf.place_dihedral(
            a, b, c, np.pi / 2, 1.0, -np.pi / 2, use_torch=True
        )
        self.assertTrue(
            torch.allclose(d, calc_d, atol=1e-6), f"Mismatched: {d} != {calc_d}"
        )

    def test_randomized(self):
        """Test using randomized values"""
        for _ in range(100):
            a, b, c, d = [
                torch.from_numpy(x)
                for x in self.rng.uniform(low=-5, high=5, size=(4, 3))
            ]
            calc_d = nerf.place_dihedral(
                a,
                b,
                c,
                angle_between((d - c).numpy(), (b - c).numpy()),
                dist_between(c.numpy(), d.numpy()),
                dihedral(a.numpy(), b.numpy(), c.numpy(), d.numpy()),
                use_torch=True,
            )
            self.assertTrue(np.allclose(d, calc_d), f"Mismatched: {d} != {calc_d}")

    def test_pytorch_vectorized(self):
        """Simple test about origin, repeated twice along batch axis"""
        n_batch = 128

        a = torch.tensor([1, 0, 0], dtype=torch.float64)
        b = torch.tensor([0, 0, 0], dtype=torch.float64)
        c = torch.tensor([0, 1, 0], dtype=torch.float64)
        d = torch.tensor([0, 1, 1], dtype=torch.float64)

        d_expanded = d.repeat(n_batch, 1)

        calc_d = nerf.place_dihedral(
            a.repeat(n_batch, 1),
            b.repeat(n_batch, 1),
            c.repeat(n_batch, 1),
            torch.tensor(np.pi / 2).repeat(n_batch, 1),
            torch.tensor(1.0).repeat(n_batch, 1),
            torch.tensor(-np.pi / 2).repeat(n_batch, 1),
            use_torch=True,
        )
        self.assertTrue(
            torch.allclose(d_expanded, calc_d, atol=1e-6),
            f"Mismatched: {d_expanded} != {calc_d}",
        )

    def test_pytorch_random_vectorized(self):
        """Test random values for vectorized computation"""
        a, b, c, d = torch.from_numpy(self.rng.random(size=(4, 1000, 3)))
        assert a.shape == b.shape == c.shape == d.shape == (1000, 3)

        angles = [
            angle_between((d_i - c_i).numpy(), (b_i - c_i))
            for b_i, d_i, c_i in zip(b, d, c)
        ]
        angles = torch.tensor(angles).unsqueeze(1)

        dists = [dist_between(c_i.numpy(), d_i.numpy()) for d_i, c_i in zip(d, c)]
        dists = torch.tensor(dists).unsqueeze(1)

        dihedrals = [
            dihedral(a_i.numpy(), b_i.numpy(), c_i.numpy(), d_i.numpy())
            for a_i, b_i, c_i, d_i in zip(a, b, c, d)
        ]
        assert len(dihedrals) == len(angles) == len(dists) == 1000
        dihedrals = torch.tensor(dihedrals).unsqueeze(1)

        calc_d = nerf.place_dihedral(a, b, c, angles, dists, dihedrals, use_torch=True)
        self.assertTrue(
            torch.allclose(d, calc_d, atol=1e-6), f"Mismatched: {d} != {calc_d}"
        )


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
