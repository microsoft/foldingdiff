"""
tests for model subparts
"""

import os, sys
import unittest

import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import modelling


class TestPositionalEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.bs = 32  # Batch size
        self.d_model = 4
        self.input_shape = (self.bs, 512, self.d_model)
        self.input = torch.randn(self.input_shape)

        self.pe = modelling.PositionalEncoding(self.d_model, max_len=512)
        self.pe.eval()  # Needed because positional encoding uses dropout

    def test_reproducibility(self):
        """
        Test that running the embedding twice should give the same embedding
        """
        x = self.pe(torch.zeros_like(self.input))
        y = self.pe(torch.zeros_like(self.input))
        self.assertTrue(torch.all(torch.isclose(x, y)), msg=f"{x} != {y}")

    def test_dimensions(self):
        """
        Test that the positional embedding is added to the correct dimension
        """
        zeros = torch.zeros_like(self.input)

        pe = self.pe(zeros)
        for i in range(1, self.bs):
            self.assertEqual(pe[i].shape, pe[0].shape)
            self.assertTrue(
                torch.all(torch.isclose(pe[i], pe[0], atol=1e-5)),
                msg=f"{pe[i]} != {pe[0]}",
            )


if __name__ == "__main__":
    unittest.main()
