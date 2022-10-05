import os
import unittest
import tempfile

import numpy as np
import torch
from transformers import BertConfig

from foldingdiff import modelling

ATOL, RTOL = 1e-6, 1e-3


class TestHuggingFaceBertModel(unittest.TestCase):
    """
    Test the BERT model for diffusion built on HuggingFace library
    """

    def setUp(self) -> None:
        self.max_seq_len = 512
        self.cfg = BertConfig(max_position_embeddings=self.max_seq_len, use_cache=False)
        self.model = modelling.BertForDiffusion(config=self.cfg)
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

    def test_batch_order_agnostic(self):
        """
        Reversing the batch order of inputs does not change output values
        """
        x = self.inputs
        with torch.no_grad():
            out = self.model(
                inputs=x,
                timestep=self.timesteps,
                attention_mask=self.attn_masks,
                position_ids=self.position_ids,
            )

        with torch.no_grad():
            rev_out = self.model(
                inputs=torch.flip(x, dims=(0,)),
                timestep=torch.flip(self.timesteps, dims=(0,)),
                attention_mask=torch.flip(self.attn_masks, dims=(0,)),
                position_ids=torch.flip(self.position_ids, dims=(0,)),
            )

        self.assertEqual(self.bs, out.shape[0])
        self.assertEqual(self.bs, rev_out.shape[0])
        self.assertTrue(
            torch.allclose(torch.flip(out, dims=(0,)), rev_out, atol=ATOL, rtol=RTOL),
            msg=f"Mismatch on reversal: {out[-2]} != {rev_out[1]}",
        )


class TestTransformerLoadingSaving(unittest.TestCase):
    """
    Test the loading, saving, and then re-loading of transformer models.
    """

    def setUp(self) -> None:
        self.orig_model_dir = os.path.join(
            os.path.dirname(__file__), "mini_model_for_testing", "results"
        )
        assert os.path.isdir(self.orig_model_dir)

    def test_saving_and_loading(self):
        """Test that we can load, save, and reload model"""
        with tempfile.TemporaryDirectory() as tempdir:
            orig_model = modelling.BertForDiffusion.from_dir(
                self.orig_model_dir, copy_to=tempdir
            )
            new_model = modelling.BertForDiffusion.from_dir(tempdir)

        # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
        for p1, p2 in zip(orig_model.parameters(), new_model.parameters()):
            self.assertAlmostEqual(p1.data.ne(p2.data).sum(), 0)

    def test_loading_without_model_checkpoint(self):
        """Test that loading ignores weights correctly"""
        with tempfile.TemporaryDirectory() as tempdir:
            orig_model = modelling.BertForDiffusion.from_dir(
                self.orig_model_dir, copy_to=tempdir
            )
            new_model = modelling.BertForDiffusion.from_dir(tempdir, load_weights=False)

        for p1, p2 in zip(orig_model.parameters(), new_model.parameters()):
            self.assertNotEqual(p1.data.ne(p2.data).sum(), 0)


class TestTransformerBaseLoadingSaving(unittest.TestCase):
    """
    Test the loading and saving and re-loading of transformer models without
    pytorch lightning
    """
    def setUp(self) -> None:
        self.orig_model_dir = os.path.join(
            os.path.dirname(__file__), "mini_model_for_testing", "results"
        )
        assert os.path.isdir(self.orig_model_dir)
    

    def test_saving_and_loading(self):
        """Test that we can load, save, and reload model"""
        with tempfile.TemporaryDirectory() as tempdir:
            orig_model = modelling.BertForDiffusionBase.from_dir(
                self.orig_model_dir, copy_to=tempdir
            )
            new_model = modelling.BertForDiffusionBase.from_dir(tempdir)

        # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
        for p1, p2 in zip(orig_model.parameters(), new_model.parameters()):
            self.assertAlmostEqual(p1.data.ne(p2.data).sum(), 0)
        
    def test_against_pl(self):
        """
        Test that loading with or without pl lightning produces the same results
        """
        with tempfile.TemporaryDirectory() as tempdir:
            nn_model = modelling.BertForDiffusionBase.from_dir(
                self.orig_model_dir, copy_to=tempdir
            )
            pl_model = modelling.BertForDiffusion.from_dir(
                self.orig_model_dir, copy_to=tempdir
            )
        for p1, p2 in zip(nn_model.parameters(), pl_model.parameters()):
            self.assertAlmostEqual(p1.data.ne(p2.data).sum(), 0)


if __name__ == "__main__":
    unittest.main()
