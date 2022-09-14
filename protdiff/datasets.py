"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

import pickle
import functools
import multiprocessing
import os
import glob
import logging
from pathlib import Path
from typing import *

from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

LOCAL_DATA_DIR = Path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
assert LOCAL_DATA_DIR.is_dir(), f"Data directory {LOCAL_DATA_DIR} does not exist"

CATH_DIR = LOCAL_DATA_DIR / "cath"
ALPHAFOLD_DIR = LOCAL_DATA_DIR / "alphafold"


import beta_schedules
from angles_and_coords import (
    canonical_distances_and_dihedrals,
    EXHAUSTIVE_ANGLES,
    EXHAUSTIVE_DISTS,
)
from custom_metrics import wrapped_mean
import utils

TRIM_STRATEGIES = Literal["leftalign", "randomcrop", "discard"]


class CathCanonicalAnglesDataset(Dataset):
    """
    Load in the dataset.

    All angles should be given between [-pi, pi]
    """

    feature_names = {
        "angles": [
            "0C:1N",
            "N:CA",
            "CA:C",
            "phi",
            "psi",
            "omega",
            "tau",
            "CA:C:1N",
            "C:1N:1CA",
        ]
    }
    feature_is_angular = {
        "angles": [False, False, False, True, True, True, True, True, True]
    }
    cache_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"cache_canonical_structures_{utils.md5_all_py_files(os.path.dirname(os.path.abspath(__file__)))}.pkl",
    )

    def __init__(
        self,
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 512,
        min_length: int = 40,  # Set to 0 to disable
        trim_strategy: TRIM_STRATEGIES = "leftalign",
        toy: int = 0,
        zero_center: bool = True,  # Center the features to have 0 mean
        use_cache: bool = True,  # Use/build cached computations of dihedrals and angles
    ) -> None:
        super().__init__()
        assert pad > min_length
        self.trim_strategy = trim_strategy
        self.pad = pad
        self.min_length = min_length

        # gather files
        fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
        assert fnames, f"No files found in {CATH_DIR}/dompdb"

        # Compute all the angles
        pfunc = functools.partial(
            canonical_distances_and_dihedrals,
            distances=EXHAUSTIVE_DISTS,
            angles=EXHAUSTIVE_ANGLES,
        )

        # self.structures should be a list of dicts with keys (angles, fname)
        # Always compute for toy; do not save
        if toy:
            if isinstance(toy, bool):
                toy = 150
            fnames = fnames[:toy]

            logging.info(f"Loading toy dataset of {toy} structures")
            struct_arrays = [pfunc(f) for f in fnames]
            self.structures = []
            for fname, s in zip(fnames, struct_arrays):
                if s is None:
                    continue
                self.structures.append({"angles": s, "fname": fname})
        elif not use_cache or not os.path.exists(self.cache_fname):
            # No cache yet or not using cache
            logging.info(
                f"Computing full dataset of {len(fnames)} with {multiprocessing.cpu_count()} threads"
            )
            # Generate dihedral angles
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            struct_arrays = pool.map(pfunc, fnames, chunksize=250)
            pool.close()
            pool.join()

            # Contains only non-null structures
            self.structures = []
            for fname, s in zip(fnames, struct_arrays):
                if s is None:
                    continue
                self.structures.append({"angles": s, "fname": fname})
            # Write the output to a file for faster loading next time
            if use_cache:
                logging.info(f"Saving full dataset to cache at {self.cache_fname}")
                with open(self.cache_fname, "wb") as sink:
                    pickle.dump(self.structures, sink)
        else:
            logging.info(f"Loading cached full dataset from {self.cache_fname}")
            with open(self.cache_fname, "rb") as source:
                self.structures = pickle.load(source)

        # If specified, remove sequences shorter than min_length
        if self.min_length:
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["angles"].shape[0] >= self.min_length
            ]
            len_delta = orig_len - len(self.structures)
            logging.info(
                f"Removing structures shorter than {self.min_length} residues excludes {len_delta}/{orig_len} --> {len(self.structures)} sequences"
            )
        if self.trim_strategy == "discard":
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["angles"].shape[0] <= self.pad
            ]
            len_delta = orig_len - len(self.structures)
            logging.info(
                f"Removing structures longer than {self.pad} produces {orig_len} - {len_delta} = {len(self.structures)} sequences"
            )

        # Split the dataset if requested. This is implemented here to maintain
        # functional parity with the original CATH dataset. Original CATH uses
        # a 80/10/10 split
        self.rng = np.random.default_rng(seed=6489)
        # Shuffle the sequences so contiguous splits acts like random splits
        self.rng.shuffle(self.structures)
        if split is not None:
            split_idx = int(len(self.structures) * 0.8)
            if split == "train":
                self.structures = self.structures[:split_idx]
            elif split == "validation":
                self.structures = self.structures[
                    split_idx : split_idx + int(len(self.structures) * 0.1)
                ]
            elif split == "test":
                self.structures = self.structures[
                    split_idx + int(len(self.structures) * 0.1) :
                ]
            else:
                raise ValueError(f"Unknown split: {split}")

            logging.info(f"Split {split} contains {len(self.structures)} structures")

        # if given, zero center the features
        self.means = None
        if zero_center:
            # Note that these angles are not yet padded
            structures_concat = np.concatenate([s["angles"] for s in self.structures])
            assert structures_concat.ndim == 2
            self.means = wrapped_mean(structures_concat, axis=0)
            assert self.means.shape == (structures_concat.shape[1],)
            # Subtract the mean and perform modulo where values are radial
            logging.info(
                f"Offsetting features {self.feature_names['angles']} by means {self.means}"
            )

        # Aggregate lengths
        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

        # for ft in self.feature_names["angles"]:
        #     idx = self.feature_names["angles"].index(ft)
        #     is_angular = self.feature_is_angular["angles"][idx]
        #     logging.info(f"Feature {ft} is angular: {is_angular}")
        #     m, v = self.get_feature_mean_var(ft)
        #     logging.info(f"Feature {ft} mean, var: {m}, {v}")

    def sample_length(self, n: int = 1) -> Union[int, List[int]]:
        """
        Sample a observed length of a sequence
        """
        assert n > 0
        if n == 1:
            l = self._length_rng.choice(self.all_lengths)
        else:
            l = self._length_rng.choice(self.all_lengths, size=n, replace=True).tolist()
        return l

    def get_masked_means(self) -> np.ndarray:
        """Return the means subset to the actual features used"""
        if self.means is None:
            return None
        return np.copy(self.means)

    @functools.cached_property
    def filenames(self) -> List[str]:
        """Return the filenames that constitute this dataset"""
        return [s["fname"] for s in self.structures]

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(
        self, index, ignore_zero_center: bool = False
    ) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        angles = self.structures[index]["angles"]

        # If given, offset the angles with mean
        if self.means is not None and not ignore_zero_center:
            assert (
                self.means.shape[0] == angles.shape[1]
            ), f"Mismatched shapes: {self.means.shape} != {angles.shape}"
            angles = angles - self.means

            # The distance features all contain a single ":"
            colon_count = np.array([c.count(":") for c in angles.columns])
            # WARNING this uses a very hacky way to find the angles
            angular_idx = np.where(colon_count != 1)[0]
            angles.iloc[:, angular_idx] = utils.modulo_with_wrapped_range(
                angles.iloc[:, angular_idx], -np.pi, np.pi
            )

        # Subset angles to ones we are actaully using as features
        angles = angles.loc[
            :, CathCanonicalAnglesDataset.feature_names["angles"]
        ].values
        assert angles is not None
        assert angles.shape[1] == len(
            CathCanonicalAnglesDataset.feature_is_angular["angles"]
        ), f"Mismatched shapes for angles: {angles.shape[1]} != {len(CathCanonicalAnglesDataset.feature_is_angular['angles'])}"

        # Replace nan values with zero
        np.nan_to_num(angles, copy=False, nan=0)

        # Create attention mask. 0 indicates masked
        l = min(self.pad, angles.shape[0])
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:l] = 1.0

        # Additionally, mask out positions that are nan
        # is_nan = np.where(np.any(np.isnan(angles), axis=1))[0]
        # attn_mask[is_nan] = 0.0  # Mask out the nan positions

        # Perform padding/trimming
        if angles.shape[0] < self.pad:
            angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif angles.shape[0] > self.pad:
            if self.trim_strategy == "leftalign":
                angles = angles[: self.pad]
            elif self.trim_strategy == "randomcrop":
                # Randomly crop the sequence to
                start_idx = self.rng.integers(0, angles.shape[0] - self.pad)
                end_idx = start_idx + self.pad
                assert end_idx < angles.shape[0]
                angles = angles[start_idx:end_idx]
                assert angles.shape[0] == self.pad
            else:
                raise ValueError(f"Unknown trim strategy: {self.trim_strategy}")

        # Create position IDs
        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        angular_idx = np.where(CathCanonicalAnglesDataset.feature_is_angular["angles"])[
            0
        ]
        assert utils.tolerant_comparison_check(
            angles[:, angular_idx], ">=", -np.pi
        ), f"Illegal value: {np.min(angles[:, angular_idx])}"
        assert utils.tolerant_comparison_check(
            angles[:, angular_idx], "<=", np.pi
        ), f"Illegal value: {np.max(angles[:, angular_idx])}"
        angles = torch.from_numpy(angles).float()

        retval = {
            "angles": angles,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
        }
        return retval

    def get_feature_mean_var(self, ft_name: str) -> Tuple[float, float]:
        """
        Return the mean and variance associated with a given feature
        """
        assert ft_name in self.feature_names["angles"], f"Unknown feature {ft_name}"
        idx = self.feature_names["angles"].index(ft_name)
        logging.info(f"Computing metrics for {ft_name} - idx {idx}")

        all_vals = []
        for i in range(len(self)):
            item = self[i]
            attn_idx = torch.where(item["attn_mask"] == 1.0)[0]
            vals = item["angles"][attn_idx, idx]
            all_vals.append(vals)
        all_vals = torch.cat(all_vals)
        assert all_vals.ndim == 1
        return torch.var_mean(all_vals)[::-1]  # Default is (var, mean)


class CathCanonicalAnglesOnlyDataset(CathCanonicalAnglesDataset):
    """
    Building on the CATH dataset, return the 3 canonical dihedrals and the 3
    non-dihedral angles. Notably, this does not return distance.
    Dihedrals: phi, psi, omega
    Non-dihedral angles: tau, CA:C:1N, C:1N:1CA
    """

    feature_names = {"angles": ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]}
    feature_is_angular = {"angles": [True, True, True, True, True, True]}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Trim out the distance in all the feature_names and feature_is_angular
        orig_features = super().feature_names["angles"].copy()
        self.feature_idx = [
            orig_features.index(ft) for ft in self.feature_names["angles"]
        ]
        logging.info(
            f"CATH canonical angles only dataset with {self.feature_names['angles']} (subset idx {self.feature_idx})"
        )

    def get_masked_means(self) -> np.ndarray:
        """Return the means subset to the actual features used"""
        if self.means is None:
            return None
        return np.copy(self.means)[self.feature_idx]

    def __getitem__(
        self, index, ignore_zero_center: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Return a dict with keys: angles, attn_mask, position_ids
        return_dict = super().__getitem__(index, ignore_zero_center=ignore_zero_center)

        # Remove the distance feature
        assert return_dict["angles"].ndim == 2
        return_dict["angles"] = return_dict["angles"][:, self.feature_idx]
        assert torch.all(
            return_dict["angles"] >= -torch.pi
        ), f"Minimum value {torch.min(return_dict['angles'])} lower than -pi"
        assert torch.all(
            return_dict["angles"] <= torch.pi
        ), f"Maximum value {torch.max(return_dict['angles'])} higher than pi"

        return return_dict


class CathCanonicalMinimalAnglesDataset(CathCanonicalAnglesOnlyDataset):
    """
    The minimal set of angles we can model and still have a reasonable protein
    reconstruction is:
    * Dihedrals: phi, psi, omega
    * Non-dihedrals: tau
    """

    feature_names = {"angles": ["phi", "psi", "omega", "tau"]}
    feature_is_angular = {"angles": [True, True, True, True]}


class AlphafoldConsecutiveAnglesDataset(Dataset):
    """
    Represent the
    """

    def __init__(
        self,
        pad: int = 512,
        shift_to_zero_twopi: bool = True,
        force_recompute_angles: bool = False,
        toy: bool = False,
    ) -> None:
        super().__init__()
        assert ALPHAFOLD_DIR.is_dir(), f"Expected AlphaFold data dir at {ALPHAFOLD_DIR}"
        self.pad = pad
        self.shift_to_zero_twpi = shift_to_zero_twopi

        # Glob for the untarred files
        pdb_files = glob.glob(os.path.join(ALPHAFOLD_DIR, "*.pdb.gz"))
        pfunc = functools.partial(
            trrosetta_angles_from_pdb, force_compute=force_recompute_angles
        )
        if toy:
            logging.info("Using toy AlphaFold dataset")
            # Reduce number of examples and disable multithreading
            pdb_files = pdb_files[:10]
            self.structures = list(map(pfunc, pdb_files))
        else:
            logging.info(f"Computing angles for {len(pdb_files)} structures")
            pool = multiprocessing.Pool()
            self.structures = pool.map(pfunc, pdb_files, chunksize=100)
            pool.close()
            pool.join()

        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

    def __str__(self) -> str:
        return f"AlphafoldConsecutiveAnglesDataset with {len(self)} examples and sequence length {np.min(self.all_lengths)}-{np.max(self.all_lengths)} padded to {self.pad}"

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not 0 <= index <= len(self):
            raise IndexError(index)

        angles = self.structures[index]["angles"]
        assert angles is not None
        l = min(self.pad, angles.shape[0])
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:l] = 1.0

        if angles.shape[0] < self.pad:
            orig_shape = angles.shape
            angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            logging.debug(f"Padded {orig_shape} --> {angles.shape}")
        elif angles.shape[0] > self.pad:
            angles = angles[: self.pad]

        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)
        return {"angles": angles, "attn_mask": attn_mask, "position_ids": position_ids}


class NoisedAnglesDataset(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__ if dset_key
    is not specified; otherwise, returns a dictionary where the item
    to noise is under dset_key

    modulo can be given as either a float or a list of floats
    """

    def __init__(
        self,
        dset: Dataset,
        dset_key: str = "angles",
        timesteps: int = 250,
        exhaustive_t: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "linear",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
    ) -> None:
        super().__init__()
        self.dset = dset
        assert hasattr(dset, "feature_names")
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        assert (
            dset_key in dset.feature_is_angular
        ), f"{dset_key} not in {dset.feature_is_angular}"
        self.n_features = len(dset.feature_is_angular[dset_key])

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)

    @property
    def feature_names(self):
        """Pass through feature names property of wrapped dset"""
        return self.dset.feature_names

    @property
    def feature_is_angular(self):
        """Pass through feature is angular property of wrapped dset"""
        return self.dset.feature_is_angular

    @property
    def pad(self):
        """Pas through the pad property of wrapped dset"""
        return self.dset.pad

    @property
    def filenames(self):
        """Pass through the filenames property of the wrapped dset"""
        return self.dset.filenames

    def sample_length(self, *args, **kwargs):
        return self.dset.sample_length(*args, **kwargs)

    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with {self.schedule}-{self.timesteps} with variance scales {self.nonangular_var_scale} and {self.angular_var_scale}"

    def __len__(self) -> int:
        if not self.exhaustive_timesteps:
            return len(self.dset)
        else:
            return int(len(self.dset) * self.timesteps)

    def plot_alpha_bar_t(self, fname: str) -> str:
        """Plot the alpha bar for each timestep"""
        fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
        vals = self.alphas_cumprod.numpy()
        ax.plot(np.arange(len(vals)), vals)
        ax.set(
            ylabel=r"$\bar \alpha_t$",
            xlabel=r"Timestep $t$",
            title=f"Alpha bar for {self.schedule} across {self.timesteps} timesteps",
        )
        fig.savefig(fname, bbox_inches="tight")
        return fname

    def sample_noise(self, vals: torch.Tensor) -> torch.Tensor:
        """
        Adaptively sample noise based on modulo. We scale only the variance because
        we want the noise to remain zero centered
        """
        # Noise is always 0 centered
        noise = torch.randn_like(vals)

        # Shapes of vals couled be (batch, seq, feat) or (seq, feat)
        # Therefore we need to index into last dimension consistently

        # Scale by provided variance scales based on angular or not
        if self.angular_var_scale != 1.0 or self.nonangular_var_scale != 1.0:
            for j in range(noise.shape[-1]):  # Last dim = feature dim
                s = (
                    self.angular_var_scale
                    if self.dset.feature_is_angular[self.dset_key][j]
                    else self.nonangular_var_scale
                )
                noise[..., j] *= s

        # Make sure that the noise doesn't run over the boundaries
        angular_idx = np.where(self.dset.feature_is_angular[self.dset_key])[0]
        noise[..., angular_idx] = utils.modulo_with_wrapped_range(
            noise[..., angular_idx], -np.pi, np.pi
        )

        return noise

    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Gets the i-th item in the dataset and adds noise
        use_t_val is useful for manually querying specific timepoints
        """
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        # Handle cases where we exhaustively loop over t
        if self.exhaustive_timesteps:
            item_index = index // self.timesteps
            assert item_index < len(self.dset)
            time_index = index % self.timesteps
            logging.debug(
                f"Exhaustive {index} -> item {item_index} at time {time_index}"
            )
            assert (
                item_index * self.timesteps + time_index == index
            ), f"Unexpected indices for {index} -- {item_index} {time_index}"
            item = self.dset.__getitem__(
                item_index, ignore_zero_center=ignore_zero_center
            )
        else:
            item = self.dset.__getitem__(index, ignore_zero_center=ignore_zero_center)

        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key].clone()
        else:
            vals = item.clone()
        assert isinstance(
            vals, torch.Tensor
        ), f"Using dset_key {self.dset_key} - expected tensor but got {type(vals)}"

        # Sample a random timepoint and add corresponding noise
        if use_t_val is not None:
            assert (
                not self.exhaustive_timesteps
            ), "Cannot use specific t in exhaustive mode"
            t_val = np.clip(np.array([use_t_val]), 0, self.timesteps - 1)
            t = torch.from_numpy(t_val).long()
        elif self.exhaustive_timesteps:
            t = torch.tensor([time_index]).long()  # list to get correct shape
        else:
            t = torch.randint(0, self.timesteps, (1,)).long()

        # Get the values for alpha and beta
        sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][t.item()]
        sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms[
            "sqrt_one_minus_alphas_cumprod"
        ][t.item()]
        # Noise is sampled within range of [-pi, pi], and optionally
        # shifted to [0, 2pi] by adding pi
        noise = self.sample_noise(vals)  # Vals passed in only for shape

        # Add noise and ensure noised vals are still in range
        noised_vals = (
            sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
        )
        assert noised_vals.shape == vals.shape, f"Unexpected shape {noised_vals.shape}"
        # The underlying vals are already shifted, and noise is already shifted
        # All we need to do is ensure we stay on the corresponding manifold
        angular_idx = np.where(self.dset.feature_is_angular[self.dset_key])[0]
        # Wrap around the correct range
        noised_vals[:, angular_idx] = utils.modulo_with_wrapped_range(
            noised_vals[:, angular_idx], -np.pi, np.pi
        )

        retval = {
            "corrupted": noised_vals,
            "t": t,
            "known_noise": noise,
        }

        # Update dictionary if wrapped dset returns dicts, else just return
        if isinstance(item, dict):
            assert item.keys().isdisjoint(retval.keys())
            item.update(retval)
            return item
        return retval


class SingleNoisedAngleDataset(NoisedAnglesDataset):
    """
    Dataset that adds noise to the angles in the dataset.
    """

    __name__ = "SingleNoisedAngleDataset"

    def __init__(
        self, use_fixed_noise: bool = False, ft_idx: int = 1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Generate a new set of noise for each instance
        # This means validation/train/test haver differnet noise
        # losses should diverge
        self.selected_index = ft_idx
        self.fixed_noise = None
        if use_fixed_noise:
            logging.warning("Using fixed noise!")
            self.fixed_noise = torch.randn((512, 4)) * torch.tensor(
                [1.0, torch.pi, torch.pi, torch.pi]
            )

    def sample_noise(self, vals):
        if self.fixed_noise is not None:
            return self.fixed_noise
        return super().sample_noise(vals)

    def __getitem__(
        self, index: int, use_t_val: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get only one angle"""
        vals = super().__getitem__(index, use_t_val)
        # Select a single angle
        for k in ["angles", "corrupted", "known_noise"]:
            assert (
                len(vals[k].shape) == 2
            ), f"Expected 2D tensor but got {vals[k].shape}"
            v = vals[k][:, self.selected_index].unsqueeze(1)
            vals[k] = v
        return vals

    def __str__(self) -> str:
        return f"{self.__name__} returning feature {self.selected_index} with fixed noise {self.fixed_noise.flatten()[:5] if self.fixed_noise is not None else None}"


class SingleNoisedBondDistanceDataset(SingleNoisedAngleDataset):
    """
    Dataset that does only the bond distance
    """

    __name__ = "SingleNoisedBondDistanceDataset"

    def __init__(self, use_fixed_noise: bool = False, *args, **kwargs) -> None:
        super().__init__(use_fixed_noise, *args, ft_idx=0, **kwargs)


class SingleNoisedAngleAndTimeDataset(SingleNoisedAngleDataset):
    """
    Datsaet that adds noise to just one angle and at only one timestep
    For extreme debugging to overfit
    """

    selected_timestep = 100

    def __getitem__(
        self, index: int, use_t_val: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        assert use_t_val is None, "Cannot use specific t for fixed timestep sampler"
        retval = super().__getitem__(index, use_t_val=self.selected_timestep)
        return retval

    def __str__(self) -> str:
        return super().__str__() + f" at timestep {self.selected_timestep}"


class SynNoisedByPositionDataset(Dataset):
    """
    SYNTHETIC NOISE FOR DEBUGGING AND TESTING

    Add noise in by time. Specifically, have the front half of the angles get
    negative noise, and the latter half get positive noise. This simple setup
    requires the model to use positional embedding effectively.

    Note that timesteps is provided only for compatibility in calling and is
    NOT actually used.
    """

    __name__ = "SynNoisedByPositionDataset"

    def __init__(
        self,
        dset: Dataset,
        dset_key: Optional[str] = None,
        var_val: float = 1.0,
        timesteps: int = 250,
        use_timesteps: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "linear",
        ft_subset: Optional[int] = 1,
        **kwargs,  # Allow passthrough since this is a debugging dataset
    ) -> None:
        super().__init__()
        self.dset = dset
        self.dset_key = dset_key
        self.ft_subset = ft_subset

        self.schedule = beta_schedule
        self.timesteps = timesteps

        # Compute beta and alpha values
        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)

        # If true, use timesteps to scale noise/original ratio
        self.use_timesteps = use_timesteps
        self.var_val = var_val
        logging.warning(f"Ignoring noiser class kwargs: {kwargs}")

    def __len__(self) -> int:
        return len(self.dset)

    def __str__(self):
        return f"{self.__name__} wrapping {self.dset} with var_val {self.var_val} selecting ft {self.ft_subset} {'WITH' if self.use_timesteps else 'WITHOUT'} timesteps"

    def sample_noise(self, vals, attn_mask) -> torch.Tensor:
        """
        Sample noise given the values to noise and attention mask
        Values ot noise are used only for their shape
        """
        # attention mask should be given in huggingface convention where
        # 1 = unmasked and 0 = masked
        seq_len = torch.sum(attn_mask)
        assert (
            seq_len <= vals.shape[0]
        ), f"Expected seq_len <= {vals.shape[0]} but got {seq_len}"

        # Sample a truncated normal distribution for both +/-
        # https://stackoverflow.com/questions/60233216/how-to-make-a-truncated-normal-distribution-in-pytorch
        pos_dist = torch.zeros_like(vals)
        nn.init.trunc_normal_(pos_dist, mean=0.0, std=self.var_val, a=0, b=torch.pi)
        assert torch.all(pos_dist >= 0.0)
        assert torch.all(pos_dist <= torch.pi)
        neg_dist = torch.zeros_like(vals)
        nn.init.trunc_normal_(neg_dist, mean=0.0, std=self.var_val, a=-torch.pi, b=0)
        assert torch.all(neg_dist >= -torch.pi)
        assert torch.all(neg_dist <= 0.0)

        # Create a noise vector where first/second half of sequence have different noise
        # Creates indices like
        # [1, 1, 1, 1]
        # [2, 2, 2, 2]
        # [3, 3, 3, 3]
        # ...
        broadcasted_indices = (
            torch.arange(vals.shape[0]).unsqueeze(1).broadcast_to(vals.shape)
        )
        noise = torch.where(broadcasted_indices < seq_len / 2, pos_dist, neg_dist)
        return noise

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.dset.__getitem__(index)
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key]
        else:
            vals = item

        if self.ft_subset is not None:
            item[self.dset_key] = vals[:, self.ft_subset].unsqueeze(1)
            vals = vals[:, self.ft_subset].unsqueeze(1)
            assert len(vals.shape) == 2
            assert vals.shape[-1] == 1

        # Sample a random timestep
        t = torch.randint(0, self.timesteps, (1,)).long()

        # Get the corrupted example
        noise = self.sample_noise(vals, item["attn_mask"])

        # Based on whether or not we are using timesteps to scale orig/noise, build
        # corrupted exapmle
        if self.use_timesteps:
            t_idx = t.item()
            sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][t_idx]
            sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms[
                "sqrt_one_minus_alphas_cumprod"
            ][t_idx]
            noised_vals = (
                sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
            )
        else:
            noised_vals = vals + noise

        # DIFFERENCE NO MODULO

        # Build output dictionary
        retval = {
            "corrupted": noised_vals,
            "t": t,
            "known_noise": noise,
        }
        if isinstance(item, dict):
            assert item.keys().isdisjoint(retval.keys())
            item.update(retval)
            return item
        raise NotImplementedError


class SynNoisedMaskedOnlyDataset(Dataset):
    """
    Synthetic dataset that noises only masked positions.

    Primarily for testing that models correctly ignore masked positions
    and NOT for training purposes. Namely, with this dataset, we should
    be able to test that model f(x) obeys
    f(angles) = f(corrupted) = f(angles + noise)
    Since the noise is only applied to masked positions by construction
    """

    def __init__(self, dset: Dataset, dset_key: str = "angles", **kwargs) -> None:
        super().__init__()
        self.dset = dset
        self.dset_key = dset_key

        logging.warning("NOT FOR TRAINING")
        logging.warning(f"Ignoring noiser class kwargs: {kwargs}")

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Add noise to masked positions only
        """
        item = self.dset.__getitem__(index)
        vals = item[self.dset_key]
        attn_mask = item["attn_mask"]

        masked_idx = torch.where(attn_mask == 0)[0]
        assert torch.all(vals[masked_idx] == 0.0)

        noise = torch.randn_like(vals)
        noise[attn_mask == 1] = 0.0  # Zero out noise in non-masked positions

        corrupted = vals + noise
        retval = {
            "corrupted": corrupted,
            "t": torch.randint(0, 250, (1,)).long(),
            "known_noise": noise,
        }
        assert item.keys().isdisjoint(retval.keys())
        item.update(retval)
        return item


class ScoreMatchingNoisedAnglesDataset(Dataset):
    """
    Add noise to perform score matching

    Based on:
    * https://arxiv.org/pdf/2206.01729.pdf
    * https://openreview.net/pdf?id=PxTIG12RRHS
    """

    sigma_min = 0.01 * np.pi
    sigma_max = np.pi
    num_ks = 5000  # Number of 2 * pi * k values to sample

    def __init__(self, dset, dset_key: Optional[str] = None) -> None:
        super().__init__()
        self.dset = dset
        self.dset_key = dset_key

    @staticmethod
    def get_sigma(t: float) -> float:
        """Return the value for sigma at time t"""
        assert 0 <= t <= 1
        return ScoreMatchingNoisedAnglesDataset.sigma_min ** (
            1.0 - t
        ) * ScoreMatchingNoisedAnglesDataset.sigma_max ** (t)

    @staticmethod
    def get_score(corr, orig, t: float):
        """
        Get the score for the given corrupted set of angles given the original set of angles
        Score corresponds to the negative log likelihood of the corrupted angles
        """
        # NOTE this is untested
        corr = (corr + np.pi) % (2 * np.pi) - np.pi
        orig = (orig + np.pi) % (2 * np.pi) - np.pi

        assert corr.shape == orig.shape
        assert 0 <= t <= 1
        sigma = ScoreMatchingNoisedAnglesDataset.get_sigma(t)
        delta = corr - orig
        delta = (delta + np.pi) % (2 * np.pi) - np.pi

        logp = 0
        for k in range(
            -ScoreMatchingNoisedAnglesDataset.num_ks,
            ScoreMatchingNoisedAnglesDataset.num_ks,
        ):
            logp += delta + 2 * np.pi * k / (2 * sigma * sigma)
        return logp

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return super().__getitem__(index)


def main():
    dset = CathCanonicalAnglesDataset(pad=128, trim_strategy="discard", use_cache=False)
    noised_dset = NoisedAnglesDataset(dset, dset_key="angles")
    print(len(noised_dset))
    print(noised_dset[0])

    # x = noised_dset[0]
    # for k, v in x.items():
    #     print(k)
    #     print(v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
