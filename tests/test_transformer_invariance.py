import os, sys
import unittest

import numpy as np
import torch
from transformers import BertConfig

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protdiff")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

import modelling

ATOL, RTOL = 1e-6, 1e-3


class TestHuggingFaceBertModel(unittest.TestCase):
    """
    Test the BERT model for diffusion built on HuggingFace library
    """

    def setUp(self) -> None:
        self.max_seq_len = 512
        self.cfg = BertConfig(max_position_embeddings=self.max_seq_len, use_cache=False)
        print(self.cfg)
        self.model = modelling.BertForDiffusion(self.cfg)
        self.model.eval()

        self.bs = 32
        rng = np.random.default_rng(6489)
        torch.random.manual_seed(6489)

        # Generate attention masks by huggingface convention
        # These should be auto-converted to the pytorch convention
        # Do not generate sequences in the last 5 so we never have attention there
        # This helps create an easy test for attention masking
        self.always_masked_residues = 5
        lengths = [
            rng.integers(100, self.max_seq_len - self.always_masked_residues)
            for _ in range(self.bs)
        ]
        self.attn_masks = torch.zeros((self.bs, self.max_seq_len))
        for i, l in enumerate(lengths):
            self.attn_masks[i][:l] = 1.0

        # Generate random timesteps
        self.timesteps = torch.from_numpy(
            rng.integers(0, 250, size=(self.bs, 1))
        ).long()

        # Generate random inputs
        self.inputs = torch.randn((self.bs, self.max_seq_len, self.model.n_inputs))

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

        # set position ids
        self.position_ids = (
            torch.arange(0, self.max_seq_len)
            .unsqueeze(0)
            .expand(self.bs, self.max_seq_len)
        )

        assert (
            self.inputs.shape[0]
            == self.timesteps.shape[0]
            == self.attn_masks.shape[0]
            == self.bs
        )
        assert self.inputs.shape[1] == self.attn_masks.shape[1]

    def test_consistency(self):
        """
        Test that given the same input the model gives the same output
        """
        x = self.model(
            inputs=self.inputs,
            timestep=self.timesteps,
            attention_mask=self.attn_masks,
            position_ids=self.position_ids,
        )
        y = self.model(
            inputs=self.inputs,
            timestep=self.timesteps,
            attention_mask=self.attn_masks,
            position_ids=self.position_ids,
        )
        self.assertTrue(torch.allclose(x, y))

    def test_noise_invariance_easy(self):
        """
        Easy test for noise invariance that focuses on the last few
        residues that should never be attended to
        """
        x = self.inputs
        with torch.no_grad():
            out = self.model(
                inputs=x,
                timestep=self.timesteps,
                attention_mask=self.attn_masks,
                position_ids=self.position_ids,
            )

        noise = torch.randn_like(x)
        noise[:, : -self.always_masked_residues] = 0.0
        # Check that there is no noise in the leading residues
        assert torch.all(noise[:, : -self.always_masked_residues] == 0.0)
        noised_x = x + noise

        with torch.no_grad():
            noised_out = self.model(
                inputs=noised_x,
                timestep=self.timesteps,
                attention_mask=self.attn_masks,
                position_ids=self.position_ids,
            )
        unmasked_idx = torch.where(self.attn_masks == 1.0)
        self.assertTrue(
            torch.allclose(
                out[unmasked_idx], noised_out[unmasked_idx], rtol=RTOL, atol=ATOL
            ),
            msg=f"Got different outputs: {out.flatten()[:5]} {noised_out.flatten()[:5]}",
        )


class TestBertDenoiserEncoderModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = modelling.BertDenoiserEncoderModel()
        self.model.eval()

        self.bs = 32
        rng = np.random.default_rng(6489)
        torch.random.manual_seed(6489)

        # Generate attention masks by huggingface convention
        # These should be auto-converted to the pytorch convention
        # Do not generate sequences in the last 5 so we never have attention there
        # This helps create an easy test for attention masking
        self.always_masked_residues = 5
        lengths = [
            rng.integers(100, self.model.max_seq_len - self.always_masked_residues)
            for _ in range(self.bs)
        ]
        self.attn_masks = torch.zeros((self.bs, self.model.max_seq_len))
        for i, l in enumerate(lengths):
            self.attn_masks[i][:l] = 1.0

        # Generate random timesteps
        self.timesteps = torch.from_numpy(
            rng.integers(0, 250, size=(self.bs, 1))
        ).long()

        # Generate random inputs
        self.inputs = torch.randn(
            (self.bs, self.model.max_seq_len, self.model.n_inputs)
        )

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

        assert (
            self.inputs.shape[0]
            == self.timesteps.shape[0]
            == self.attn_masks.shape[0]
            == self.bs
        )
        assert self.inputs.shape[1] == self.attn_masks.shape[1]

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

        # Shuffle the order of the inputs and run them through the model again, expect same output
        # https://pytorch.org/docs/stable/generated/torch.randperm.html
        idx = torch.randperm(x.shape[0])
        assert idx.shape[0] == x.shape[0]
        with torch.no_grad():
            shuffled_out = self.model(
                x=x[idx], timestep=self.timesteps[idx], attn_mask=self.attn_masks[idx]
            )
        # Shuffle the known outputs to match
        out_reordered = out[idx]
        self.assertTrue(
            torch.allclose(out_reordered, shuffled_out, atol=ATOL, rtol=RTOL),
            msg=f"Got different outputs: {out.flatten()[:5]} {shuffled_out.flatten()[:5]}",
        )

    def test_batch_order_consistency_reversed(self):
        """
        Test that reversing the batch order of inputs does not change output
        """
        x = self.inputs
        with torch.no_grad():
            out = self.model(x=x, timestep=self.timesteps, attn_mask=self.attn_masks)

        with torch.no_grad():
            rev_out = self.model(
                x=torch.flip(x, dims=(0,)),
                timestep=torch.flip(self.timesteps, dims=(0,)),
                attn_mask=torch.flip(self.attn_masks, dims=(0,)),
            )

        self.assertEqual(self.bs, out.shape[0])
        self.assertEqual(self.bs, rev_out.shape[0])
        self.assertTrue(
            torch.allclose(torch.flip(out, dims=(0,)), rev_out, atol=ATOL, rtol=RTOL),
            msg=f"Mismatch on reversal: {out[-2]} != {rev_out[1]}",
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
        # pytorch masked indicies are indicated by True
        conv_masked_indices = torch.where(converted_mask)
        for i, j in zip(orig_masked_indices, conv_masked_indices):
            self.assertTrue(torch.all(i == j))

    def test_noise_invariance_easy(self):
        """
        Easy test for noise invariance that focuses on the last few
        residues that should never be attended to
        """
        x = self.inputs
        with torch.no_grad():
            out = self.model(x=x, timestep=self.timesteps, attn_mask=self.attn_masks)

        noise = torch.randn_like(x)
        noise[:, : -self.always_masked_residues] = 0.0
        # Check that there is no noise in the leading residues
        assert torch.all(noise[:, : -self.always_masked_residues] == 0.0)
        noised_x = x + noise

        with torch.no_grad():
            noised_out = self.model(
                x=noised_x, timestep=self.timesteps, attn_mask=self.attn_masks,
            )

        unmasked_idx = torch.where(self.attn_masks == 1.0)
        self.assertTrue(
            torch.allclose(out[unmasked_idx], noised_out[unmasked_idx]),
            msg=f"Got different outputs: {out.flatten()[:5]} {noised_out.flatten()[:5]}",
        )

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
        
        unmasked_idx = torch.where(self.attn_masks == 1.0)
        self.assertTrue(
            torch.allclose(out[unmasked_idx], noised_out[unmasked_idx]),
            msg=f"Got different outputs: {out.flatten()[:5]} {noised_out.flatten()[:5]}",
        )


if __name__ == "__main__":
    unittest.main()
