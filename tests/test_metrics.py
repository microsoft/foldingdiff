import os, sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import custom_metrics as cm


class TestKLFromEmpirical(unittest.TestCase):
    """
    Test for KL divergence calculation
    """

    def test_two_gaussians(self):
        """The KL divergence between two Gaussians should be infinity"""
        # https://stats.stackexchange.com/questions/362860/kl-divergence-between-which-distributions-could-be-infinity
        rng = np.random.default_rng(seed=6789)
        u = rng.normal(loc=0.0, scale=1.0, size=1000)
        v = rng.normal(loc=0.0, scale=1.0, size=1000)

        kl = cm.kl_from_empirical(u, v)
        self.assertEqual(np.inf, kl)

    def test_slightly_diff_gaussians(self):
        """The KL divergence between these should be small"""
        rng = np.random.default_rng(seed=6789)
        u = rng.normal(loc=0.0, scale=2.0, size=1000)
        v = rng.normal(loc=0.0, scale=1.0, size=1000)
        w = rng.normal(loc=0.0, scale=0.5, size=1000)

        uv = cm.kl_from_empirical(v, u)
        uw = cm.kl_from_empirical(w, u)
        self.assertLess(uv, uw)

    def test_nonoverlapping(self):
        """KL divergence between non overlapping distributions is inf"""
        rng = np.random.default_rng(seed=6789)
        u = rng.normal(loc=0.0, scale=1.0, size=1000)
        v = rng.normal(loc=10.0, scale=1.0, size=1000)

        kl = cm.kl_from_empirical(u, v)
        self.assertEqual(np.inf, kl)
