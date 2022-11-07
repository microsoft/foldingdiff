"""
tests for model subparts
"""

import unittest

import torch

from foldingdiff import modelling


class TestPositionalEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.bs = 32  # Batch size
        self.d_model = 4
        self.seq_len = 512
        self.input_shape = (self.bs, self.seq_len, self.d_model)
        self.input = torch.randn(self.input_shape)

        self.pe = modelling.PositionalEncoding(self.d_model, max_len=self.seq_len)
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
            # Check that across the batch, every example gets the same embedding
            self.assertTrue(
                torch.all(torch.isclose(pe[i], pe[0])),
                msg=f"{pe[i]} != {pe[0]}",
            )
            # Check that across the sequences, each position gets the same embedding
            for j in range(self.seq_len):
                self.assertTrue(torch.all(torch.isclose(pe[i][j], pe[0][j])))


class TestGaussianFourierProjection(unittest.TestCase):
    """
    Tests for the Gaussian Fourier projection time handling code
    """

    def setUp(self) -> None:
        torch.random.manual_seed(6489)

        self.bs = 32  # Batch size
        self.d_model = 4
        self.seq_len = 512
        self.input_shape = (self.bs, self.seq_len, self.d_model)
        self.input = torch.randn(self.input_shape)
        self.timesteps = torch.randint(low=0, high=250, size=(self.bs, 1))

        self.embedder = modelling.GaussianFourierProjection(embed_dim=self.d_model)
        self.embedder.eval()  # Needed because positional encoding uses dropout

    def test_reproducibility(self):
        """Test that running code twice gets same result"""
        x = self.embedder(self.timesteps)
        y = self.embedder(self.timesteps)
        self.assertTrue(torch.all(torch.isclose(x, y)), msg=f"{x} != {y}")

    def test_permutation(self):
        """
        Test that permuting the input permutes the output predictably
        """
        x = self.embedder(self.timesteps)

        idx = torch.randperm(self.bs)
        x_permuted = self.embedder(self.timesteps[idx])

        self.assertTrue(torch.all(torch.isclose(x[idx], x_permuted)))

    def test_uniqueness(self):
        """
        Test that across a range of inputs, each output is unique
        """
        x = torch.arange(0, 1000)
        e = self.embedder(x)
        for i in range(1000):
            for j in range(1000):
                if i == j:
                    continue
                self.assertFalse(torch.allclose(e[i], e[j]))


if __name__ == "__main__":
    unittest.main()
