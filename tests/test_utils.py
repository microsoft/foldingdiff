import os, sys
import unittest

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import utils


class TestModuloWithWrappedRange(unittest.TestCase):
    """Test the modulo with wrapped range"""

    def test_simple(self):
        """Test a hand-engineered example"""
        x = utils.modulo_with_wrapped_range(3, -2, 2)
        self.assertEqual(-1, x)

    def test_simple2(self):
        """Another hand engineered example"""
        x = utils.modulo_with_wrapped_range(5, -2, 2)
        self.assertEqual(1, x)

    def test_negative(self):
        """Test a case of a negative value"""
        x = utils.modulo_with_wrapped_range(-1, -2, 2)
        self.assertEqual(-1, x)

    def test_negative_wrapped(self):
        """Test a case where a negative value needs wrapping"""
        x = utils.modulo_with_wrapped_range(-3, -2, 2)
        self.assertEqual(1, x)

    def test_positive_range(self):
        """Test a case where the range is nonnegative"""
        x = utils.modulo_with_wrapped_range(3, 0, 4)
        self.assertEqual(3, x)
    
    def test_positive_range_wrapped(self):
        """Test a case where the range is nonnegative and needs wrapping"""
        x = utils.modulo_with_wrapped_range(5, 0, 4)
        self.assertEqual(1, x)
    
    def test_positive_range_negative_value(self):
        """Test that a negative value to a nonnegative range is wrapped"""
        x = utils.modulo_with_wrapped_range(-1, 0, 4)
        self.assertEqual(3, x)
    
    def test_arr(self):
        """Test with an array"""
        x = utils.modulo_with_wrapped_range(np.array([2, -2]), -2, 2)
        self.assertTrue(np.allclose(np.array([-2, -2]), x))
    
    def test_arr2(self):
        """Test with another array"""
        x = utils.modulo_with_wrapped_range(np.array([1, -3]), -2, 2)
        self.assertTrue(np.allclose(np.array([1, 1]), x))


if __name__ == "__main__":
    unittest.main()
