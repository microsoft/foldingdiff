"""
Test the TMalign wrapper
"""

import os
import unittest

from foldingdiff import tmalign


class TestTMalign(unittest.TestCase):
    """
    Test the TMalign wrapper
    """

    def setUp(self) -> None:
        self.fname1 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "7PFL.pdb"
        )
        self.fname2 = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "7ZYA.pdb"
        )
        assert os.path.isfile(self.fname1)
        assert os.path.isfile(self.fname2)

    def test_reproducibility(self):
        """Test that the same file run twice gives consistent results"""
        x = tmalign.run_tmalign(self.fname1, self.fname2)
        y = tmalign.run_tmalign(self.fname1, self.fname2)
        self.assertAlmostEqual(x, y)

    def test_self_is_1(self):
        """Test that comparing a file to itself should yield a perfect score of 1"""
        x = tmalign.run_tmalign(self.fname1, self.fname1)
        self.assertAlmostEqual(x, 1.0)
    
    def test_nonself_less_than_1(self):
        """Test that comparing against a different structure yields a value < 1"""
        x = tmalign.run_tmalign(self.fname1, self.fname2)
        self.assertLess(x, 1.0)

if __name__ == "__main__":
    unittest.main()
