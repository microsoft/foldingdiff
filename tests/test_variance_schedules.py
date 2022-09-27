"""
Tests for variance schedules
"""

import unittest

import torch

from foldingdiff import beta_schedules


class TestLinearVarianceSchedule(unittest.TestCase):
    def test_strictly_increasing(self):
        """
        Test that betas should be strictly increasing
        """
        betas = beta_schedules.linear_beta_schedule(100)
        deltas = betas[1:] - betas[:-1]
        self.assertTrue(torch.all(deltas > 0))


class TestCosineVarianceSchedule(unittest.TestCase):
    def test_strictly_increasing(self):
        """
        Test that betas should be strictly increasing
        """
        betas = beta_schedules.cosine_beta_schedule(100)
        deltas = betas[1:] - betas[:-1]
        self.assertTrue(torch.all(deltas > 0))


class TestQuadraticrVarianceSchedule(unittest.TestCase):
    def test_strictly_increasing(self):
        """
        Test that betas should be strictly increasing
        """
        betas = beta_schedules.quadratic_beta_schedule(100)
        deltas = betas[1:] - betas[:-1]
        self.assertTrue(torch.all(deltas > 0))


if __name__ == "__main__":
    unittest.main()
