import os, sys
import unittest

import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import modelling


class TestBertDenoiserEncoderModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = modelling.BertDenoiserEncoderModel()
        self.model.eval()

        bs = 32
        rng = np.random.default_rng(6489)
        torch.random.manual_seed(6489)

        # Generate attention masks by huggingface convention
        # These should be auto-converted to the pytorch convention
        lengths = [rng.integers(100, self.model.max_seq_len) for _ in range(bs)]
        self.attn_masks = torch.zeros((bs, self.model.max_seq_len))
        for i, l in enumerate(lengths):
            self.attn_masks[i][:l] = 1.0

        # Generate random timesteps
        self.timesteps = torch.from_numpy(rng.integers(0, 250, size=(bs, 1))).long()

        # Generate random inputs
        self.inputs = torch.randn((bs, self.model.max_seq_len, self.model.n_inputs))

        # Generate noise vectors that correspond to masked positions
        unmask_positions = torch.where(self.attn_masks == 1.0)
        self.noise_on_masked = torch.randn_like(self.inputs)
        # zero out the positions that are not masked so we do NOT noise them
        self.noise_on_masked[unmask_positions] = 0.0
        for i, l in enumerate(lengths):
            assert torch.all(
                torch.isclose(self.noise_on_masked[i][:l], torch.tensor(0.0))
            )

        # Inputs with noise on masked positions
        self.inputs_with_noise_on_mask = self.inputs + self.noise_on_masked
        for i, l in enumerate(lengths):
            # Check that the unmaske indices are unmodified
            assert torch.all(
                torch.isclose(self.inputs_with_noise_on_mask[i][:l], self.inputs[i][:l])
            )

    def test_consistency(self):
        """
        Test that given the same input the model gives the same output
        """
        x = self.model(
            x=self.inputs, timestep=self.timesteps, attn_mask=self.attn_masks
        )
        y = self.model(
            x=self.inputs, timestep=self.timesteps, attn_mask=self.attn_masks
        )
        self.assertTrue(torch.allclose(x, y))

    def test_batch_order_consistency(self):
        """
        Test that the model is invariant to the order of inputs in a batch
        """
        # Run the inputs through as a "baseline" set of values
        x = self.inputs
        with torch.no_grad():
            out = self.model(x=x, timestep=self.timesteps, attn_mask=self.attn_masks)

        # Reverse the order of the inputs and run them through the model again, expect same output
        idx = torch.randperm(x.shape[0])
        with torch.no_grad():
            shuffled_out = self.model(
                x=x[idx], timestep=self.timesteps[idx], attn_mask=self.attn_masks[idx]
            )
        # Shuffle the known outputs to match
        out_reordered = out[idx]
        self.assertTrue(
            torch.allclose(out_reordered, shuffled_out),
            msg=f"Got different outputs: {out.flatten()[:5]} {shuffled_out.flatten()[:5]}",
        )

    def test_attn_mask_reformat(self):
        """
        Test that the mask format is detected and converted to correct
        PyTorch native format. Specifically, huggingface gives masked
        positions as 0.0, pytorch expects masked positions as True
        """
        converted_mask = self.model.ensure_mask_fmt(self.attn_masks)
        # huggingface masked indices are indicated by 0.
        orig_masked_indices = torch.where(self.attn_masks == 0.0)
        conv_masked_indices = torch.where(converted_mask)
        for i, j in zip(orig_masked_indices, conv_masked_indices):
            self.assertTrue(torch.all(i == j))

    def test_noise_invariance(self):
        """
        Test that noising masked positions should not affect output
        """
        # Run the inputs through as a "baseline" set of values
        x = self.inputs
        with torch.no_grad():
            out = self.model(x=x, timestep=self.timesteps, attn_mask=self.attn_masks)

        # Noise the inputs and run them through the model again, expect same output
        noised_x = x + self.noise_on_masked
        with torch.no_grad():
            noised_out = self.model(
                x=noised_x, timestep=self.timesteps, attn_mask=self.attn_masks,
            )
        self.assertTrue(
            torch.allclose(out, noised_out),
            msg=f"Got different outputs: {out.flatten()[:5]} {noised_out.flatten()[:5]}",
        )


if __name__ == "__main__":
    unittest.main()
