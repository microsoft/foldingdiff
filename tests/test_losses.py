import unittest

import numpy as np
import torch

from foldingdiff import losses


class TestRadianSmoothL1Loss(unittest.TestCase):
    def test_easy(self):
        """
        Easy test of basic wrapping functionality
        """
        l = losses.radian_smooth_l1_loss(torch.tensor(0.1), 2 * torch.pi, beta=1.0)
        self.assertAlmostEqual(0.0050, l.item())

    def test_rounding(self):
        """
        Test that rounding about unit circle works. Here, we test a case
        comparing 0 + 0.1 and 0 - 0.1 and expect the result to be 0.2 ** 2 / 2 = 0.02
        """
        l = losses.radian_smooth_l1_loss(
            torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1), beta=1.0
        )
        self.assertAlmostEqual(0.02, l.item())

    def test_no_rounding(self):
        """
        Test that loss produces correct results when rounding is not necessary
        """
        l = losses.radian_smooth_l1_loss(
            torch.tensor(0.0), torch.tensor(3.14), beta=1.0
        )
        self.assertAlmostEqual(2.64, l.item(), places=5)

    def test_double_positive_rounding(self):
        """
        Test when we have two values, both positive, one requiring rounding
        """
        l = losses.radian_smooth_l1_loss(torch.tensor(2.0), torch.tensor(4.0), beta=1.0)
        self.assertAlmostEqual(1.5, l.item(), places=5)

    def test_neg_pos(self):
        """
        Test that with two values, one negative, and one positive (past one turn) we
        get the correct result
        """
        l = losses.radian_smooth_l1_loss(
            torch.tensor(-0.1), torch.tensor(torch.pi + 2), beta=0.1
        )
        self.assertAlmostEqual(0.991593, l.item(), places=5)

    def test_neg_pos_2(self):
        """
        Another test that a negative and a positive value are correct
        """
        l = losses.radian_smooth_l1_loss(
            torch.tensor(0.5), torch.tensor(-torch.pi), beta=0.1
        )  # 3.14 - 0.5 = 2.64 - 0.1 / 2 = 2.59
        self.assertAlmostEqual(2.591593, l.item(), places=5)

    def test_zeros(self):
        """
        test that a difference of 2pi is always returned as 0
        """
        for i in range(-10, 10):
            l = losses.radian_smooth_l1_loss(
                torch.tensor(0),
                torch.tensor(i * 2 * np.pi),
            )
            self.assertAlmostEqual(0, l.item(), places=5)

    def test_loop_neg_neg(self):
        """
        Test that two negative values always produce the same result
        no matter how they are shifted
        """
        x = torch.tensor(-0.1)
        y = torch.tensor(-1.0)
        l = losses.radian_smooth_l1_loss(x, y, beta=0.1)
        # 0.9 difference -> 0.5 * 0.9 * 0.9 / 0.1 = 4.05
        for i in range(-10, 10):
            for j in range(-10, 10):
                l = losses.radian_smooth_l1_loss(
                    x + i * 2 * np.pi, y + j * 2 * np.pi, beta=0.1
                )
                self.assertAlmostEqual(0.85, l.item(), places=4)

    def test_loop_neg_pos(self):
        x = torch.tensor(-0.1)
        y = torch.tensor(1.0)
        l = losses.radian_smooth_l1_loss(x, y, beta=0.1)
        # 0.9 difference -> 0.5 * 0.9 * 0.9 / 0.1 = 4.05
        for i in range(-10, 10):
            for j in range(-10, 10):
                l = losses.radian_smooth_l1_loss(
                    x + i * 2 * np.pi, y + j * 2 * np.pi, beta=0.1
                )
                self.assertAlmostEqual(1.05, l.item(), places=4)

    def test_loop_pos_neg(self):
        x = torch.tensor(0.1)
        y = torch.tensor(-1.0)
        l = losses.radian_smooth_l1_loss(x, y, beta=0.1)
        # 0.9 difference -> 0.5 * 0.9 * 0.9 / 0.1 = 4.05
        for i in range(-10, 10):
            for j in range(-10, 10):
                l = losses.radian_smooth_l1_loss(
                    x + i * 2 * np.pi, y + j * 2 * np.pi, beta=0.1
                )
                self.assertAlmostEqual(1.05, l.item(), places=4)

    def test_loop_pos_pos(self):
        x = torch.tensor(0.1)
        y = torch.tensor(1.0)
        l = losses.radian_smooth_l1_loss(x, y, beta=0.1)
        # 0.9 difference -> 0.5 * 0.9 * 0.9 / 0.1 = 4.05
        for i in range(-10, 10):
            for j in range(-10, 10):
                l = losses.radian_smooth_l1_loss(
                    x + i * 2 * np.pi, y + j * 2 * np.pi, beta=0.1
                )
                self.assertAlmostEqual(0.85, l.item(), places=4)

    def test_symmetric(self):
        """
        Test that loss is symmetric
        """
        rng = np.random.default_rng(6489)
        for _ in range(100):
            x, y = rng.uniform(low=-200 * np.pi, high=200 * np.pi, size=2)
            x, y = torch.tensor(x), torch.tensor(y)
            i = losses.radian_smooth_l1_loss(x, y)
            j = losses.radian_smooth_l1_loss(y, x)
            self.assertAlmostEqual(i.item(), j.item())

    def test_ex(self):
        """
        Specific example
        """
        l = losses.radian_smooth_l1_loss(
            torch.tensor(-17.0466), torch.tensor(-1.3888), beta=0.1
        )
        self.assertAlmostEqual(3.04143, l.item(), places=5)


class TestPairwiseDistLoss(unittest.TestCase):
    """
    Tests for pairwise distance loss
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(1234)
        self.input = torch.from_numpy(self.rng.random(size=(128, 48, 3)))
        self.target = torch.from_numpy(self.rng.random(size=(128, 48, 3)))
        self.lengths = torch.from_numpy(self.rng.integers(low=1, high=48, size=(128,)))

    def test_shift_invariance(self):
        """Test that loss does not change under shift"""
        l = losses.pairwise_dist_loss(self.input, self.target)
        l_shift = losses.pairwise_dist_loss(self.input + 1, self.target + 100)
        self.assertAlmostEqual(l.item(), l_shift.item(), places=5)

    def test_length_masking_eq(self):
        """Test that loss correctly ignores values beyond the length mask"""
        l_ref = losses.pairwise_dist_loss(self.input, self.target, self.lengths)
        input_mutated = self.input.clone()
        for i, l in enumerate(self.lengths):
            input_mutated[i, l:] = -100.0  # Mutate the input
        l_masked = losses.pairwise_dist_loss(input_mutated, self.target, self.lengths)
        self.assertAlmostEqual(l_ref.item(), l_masked.item(), places=5)

    def test_length_masking_neq(self):
        """Test that mutating values within the length mask changes loss"""
        l_ref = losses.pairwise_dist_loss(self.input, self.target, self.lengths)
        input_mutated = self.input.clone()
        for i, l in enumerate(self.lengths):
            idx = self.rng.integers(low=0, high=l, size=1)
            input_mutated[i, idx] = -99.0
        l_mut = losses.pairwise_dist_loss(input_mutated, self.target, self.lengths)
        self.assertNotAlmostEqual(l_ref.item(), l_mut.item(), places=5)

    def test_reduce_on_closer_match(self):
        """Test that pairwise loss goes down when we have closer match"""

        l_ref = losses.pairwise_dist_loss(self.input, self.target, self.lengths)
        # Adjust target to be somewhat closer to input
        target_mutated = self.target.clone()
        for i, l in enumerate(self.lengths):
            idx = self.rng.integers(low=0, high=l, size=1)
            target_mutated[i, idx] = self.input[i, idx]
        l_new = losses.pairwise_dist_loss(self.input, target_mutated, self.lengths)
        self.assertLess(l_new.item(), l_ref.item())
    
    def test_zero_on_identical(self):
        """Test that pairwise loss is zero when inputs are identical (up to shift)"""
        l_zero = losses.pairwise_dist_loss(self.input, self.input + 99.9, self.lengths)
        self.assertAlmostEqual(l_zero.item(), 0.0, places=5)
    
    def test_symmetric(self):
        """Test that pairwise loss is symmetric"""
        l = losses.pairwise_dist_loss(self.input, self.target, self.lengths)
        l_sym = losses.pairwise_dist_loss(self.target, self.input, self.lengths)
        self.assertAlmostEqual(l.item(), l_sym.item(), places=5)


if __name__ == "__main__":
    unittest.main()
