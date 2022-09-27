import unittest

import numpy as np

from foldingdiff import custom_metrics as cm


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


class TestWrappedMean(unittest.TestCase):
    """Test for the wrapped mean function"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=6489)
        self.rad2deg = lambda x: x * 180 / np.pi
        self.deg2rad = lambda x: x * np.pi / 180

    def test_simple(self):
        """Test a hand-engineered example"""
        true_mean = 170
        x = np.array([true_mean - 30, true_mean + 30])
        x_rad = self.deg2rad(x)
        m = cm.wrapped_mean(x_rad)
        m_deg = self.rad2deg(m)
        self.assertAlmostEqual(true_mean, m_deg, places=2)

    def test_positive(self):
        """Simple test"""
        x = self.rng.normal(loc=3.0, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, 3.0, places=2)

    def test_negative(self):
        """Test that wrapping a negative mean works"""
        x = self.rng.normal(loc=-3.0, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, -3.0, places=2)

    def test_zero(self):
        """Test that a zero mean is still correctly handled"""
        x = self.rng.normal(loc=0.0, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, 0.0, places=2)

    def test_positive_unwrapped(self):
        """Test positive values that don't actually require wrapping"""
        x = self.rng.normal(loc=0.5, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, 0.5, places=2)

    def test_negative_unwrapped(self):
        """Test negative values don't actually require wrapping"""
        x = self.rng.normal(loc=-0.5, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, -0.5, places=2)

    def test_with_nan(self):
        """Test that nan values do not affect calculation"""
        x = self.rng.normal(loc=-0.5, scale=0.25, size=100000)
        m = cm.wrapped_mean(x)

        x[:100] = np.nan
        m_with_nan = cm.wrapped_mean(x)
        self.assertAlmostEqual(m, m_with_nan, places=2)
