"""
Tests for variance schedules
"""

import os, sys
import unittest

import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import beta_schedules


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
