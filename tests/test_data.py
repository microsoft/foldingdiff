"""
Unit tests to test data loaders. These primarily check that the data loaders return values
with expected shapes and ranges.
"""

import unittest

import numpy as np
import torch

from foldingdiff import datasets, utils


class TestCathCanonical(unittest.TestCase):
    """
    Tests for the cath canonical angles dataset (i.e., not the trRosetta ones)
    """

    def setUp(self) -> None:
        # Setup the dataset
        self.pad = 512
        # Use caching to avoid recomputing the whole dataset each time
        self.dset = datasets.CathCanonicalAnglesDataset(pad=self.pad, use_cache=True)

    def test_return_keys(self):
        """Test that returned dictionary has expected keys"""
        d = self.dset[0]
        self.assertEqual(
            set(d.keys()),
            set(["angles", "coords", "position_ids", "attn_mask", "lengths"]),
        )

    def test_num_feature(self):
        """Test that we have the expected number of features"""
        d = self.dset[0]
        self.assertEqual(d["angles"].shape[1], 9)

    def test_shapes(self):
        """Test that the returned tensors have expected shapes"""
        d = self.dset[1]
        self.assertEqual(
            d["angles"].shape, (self.pad, len(self.dset.feature_names["angles"]))
        )
        self.assertEqual(d["position_ids"].shape, (self.pad,))
        self.assertEqual(d["attn_mask"].shape, (self.pad,))

    def test_angles(self):
        """Test that angles do not fall outside of -pi and pi range"""
        d = self.dset[2]
        angular_idx = np.where(self.dset.feature_is_angular["angles"])[0]
        self.assertTrue(np.all(d["angles"].numpy()[..., angular_idx] >= -np.pi))
        self.assertTrue(np.all(d["angles"].numpy()[..., angular_idx] <= np.pi))


class TestCathCanonicalAnglesOnly(unittest.TestCase):
    """
    Tests for the CATH canonical angles only dataset (i.e. no distance returned)
    """

    def setUp(self) -> None:
        self.pad = 512
        self.dset = datasets.CathCanonicalAnglesOnlyDataset(
            pad=self.pad, zero_center=False
        )
        self.zero_centered_dataset = datasets.CathCanonicalAnglesOnlyDataset(
            pad=self.pad, zero_center=True
        )

    def test_return_keys(self):
        """Test that returned dictionary has expected keys"""
        d = self.dset[0]
        self.assertEqual(
            set(d.keys()),
            set(["angles", "position_ids", "attn_mask", "coords", "lengths"]),
        )

    def test_num_features(self):
        """Test that we return the expected number of features and have correctly removed distance"""
        d = self.dset[1]
        self.assertEqual(d["angles"].shape[1], 6)

    def test_all_angular(self):
        """Test that the dataset is all angular features and that this is properly registered"""
        self.assertTrue(all(self.dset.feature_is_angular["angles"]))

    def test_shapes(self):
        """Test that the returned tensors have expected shapes"""
        d = self.dset[1]
        self.assertEqual(
            d["angles"].shape, (self.pad, len(self.dset.feature_names["angles"]))
        )
        self.assertEqual(d["position_ids"].shape, (self.pad,))
        self.assertEqual(d["attn_mask"].shape, (self.pad,))

    def test_angular_range(self):
        """Test that the returned angles are all between -pi and pi"""
        d = self.dset[5]
        self.assertTrue(np.all(d["angles"].numpy() >= -np.pi))
        self.assertTrue(np.all(d["angles"].numpy() <= np.pi))

    def test_repeated_init(self):
        """Test that repeatedly intializing does not break anything"""
        # This can happy because of the way we define subclasses
        dset1 = datasets.CathCanonicalAnglesOnlyDataset(pad=self.pad)
        dset2 = datasets.CathCanonicalAnglesOnlyDataset(pad=self.pad)
        self.assertTrue(
            all(
                [
                    a == b
                    for a, b in zip(
                        dset1.feature_names["angles"], dset2.feature_names["angles"]
                    )
                ]
            )
        )

    def test_repeated_query(self):
        """Test that repeated query is consistent"""
        x1 = self.dset[0]
        x2 = self.dset[0]

        for k1 in x1.keys():
            v1 = x1[k1]
            v2 = x2[k1]
            self.assertTrue(torch.allclose(v1, v2))

    def test_repeated_query_zero_center(self):
        """Test that repeated query is consistent if we are using zero centering"""
        x1 = self.zero_centered_dataset[0]
        x2 = self.zero_centered_dataset[0]

        for k1 in x1.keys():
            v1 = x1[k1]
            v2 = x2[k1]
            self.assertTrue(torch.allclose(v1, v2))


class TestNoisedDataset(unittest.TestCase):
    """
    Tests for noised angles dataset
    """

    def setUp(self) -> None:
        self.pad = 128
        self.clean_dset = datasets.CathCanonicalAnglesOnlyDataset(
            pad=self.pad, zero_center=True, trim_strategy="leftalign"
        )
        self.noised_dset = datasets.NoisedAnglesDataset(self.clean_dset)

    def test_repeated_query(self):
        """Test that repeating a query results in the same *unnoised* start"""
        x = self.noised_dset[1]["angles"]
        y = self.noised_dset[1]["angles"]
        self.assertTrue(torch.allclose(x, y))

    def test_angles_reconstructed(self):
        """Test that subtracting noise from corrupted angles (with constant scaling) recovers original angles"""
        d = self.noised_dset[3]
        noised_angles = d["corrupted"]
        orig_angles = d["angles"]
        noise = d["known_noise"]

        recovered = (noised_angles - d["sqrt_one_minus_alphas_cumprod_t"] * noise) / d[
            "sqrt_alphas_cumprod_t"
        ]
        recovered = utils.modulo_with_wrapped_range(recovered, -np.pi, np.pi)
        self.assertTrue(torch.allclose(recovered, orig_angles, atol=1e-5))
