"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

import functools
import multiprocessing
import os, sys
import logging
import json
from typing import *
import torch

from tqdm.auto import tqdm

import numpy as np
from torch.utils.data import Dataset

CATH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cath"
)
assert os.path.isdir(CATH_DIR), f"Expected cath data at {CATH_DIR}"

from sequence_models import pdb_utils
import beta_schedules
import utils


class CathConsecutiveAnglesDataset(Dataset):
    """
    Represent proteins as their constituent angles instead of 3D coordinates

    The three angles phi, psi, and omega determine the backbone structure.
    Omega is typically fixed ~180 degrees in most cases.

    By default the angles are given in range of [-pi, pi] but we want to shift
    these to [0, 2pi] so we can have easier modulo math. This behavior can be
    toggled in shift_to_zero_twopi

    Useful reading:
    - https://proteinstructures.com/structure/ramachandran-plot/
    - https://foldit.fandom.com/wiki/Backbone_angle
    - http://www.cryst.bbk.ac.uk/PPS95/course/9_quaternary/3_geometry/torsion.html
    - https://swissmodel.expasy.org/course/text/chapter1.htm
    - https://www.nature.com/articles/s41598-020-76317-6
    - https://userguide.mdanalysis.org/1.1.1/examples/analysis/structure/dihedrals.html
    """

    def __init__(
        self,
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 512,
        shift_to_zero_twopi: bool = True,
        toy: bool = False,
    ) -> None:
        super().__init__()

        self.pad = pad
        self.shift_to_zero_twopi = shift_to_zero_twopi
        # json list file -- each line is a json
        data_file = os.path.join(CATH_DIR, "chain_set.jsonl")
        assert os.path.isfile(data_file)
        self.structures = []
        with open(data_file) as source:
            for i, line in enumerate(source):
                structure = json.loads(line.strip())
                self.structures.append(structure)
                if toy and i > 150:
                    break

        # Get data split if given
        self.split = split
        if split is not None:
            splitfile = os.path.join(CATH_DIR, "chain_set_splits.json")
            with open(splitfile) as source:
                data_splits = json.load(source)
            assert split in data_splits.keys(), f"Unrecognized split: {split}"
            # Subset self.structures to only the given names
            orig_len = len(self.structures)
            self.structures = [
                s for s in self.structures if s["name"] in set(data_splits[self.split])
            ]
            logging.info(
                f"Split {self.split} contains {len(self.structures)}/{orig_len} examples"
            )

        # Generate angles in parallel and attach them to corresponding structures
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pfunc = functools.partial(
            coords_to_angles, shift_angles_positive=self.shift_to_zero_twopi
        )
        angles = pool.map(pfunc, [d["coords"] for d in self.structures], chunksize=250)
        pool.close()
        pool.join()
        for s, a in zip(self.structures, angles):
            s["angles"] = a

        # Remove items with nan in angles/structures
        orig_count = len(self.structures)
        self.structures = [s for s in self.structures if s["angles"] is not None]
        new_count = len(self.structures)
        logging.info(f"Removed structures with nan {orig_count} -> {new_count}")

        # Aggregate the lengths
        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

    def sample_length(self) -> int:
        """
        Sample a observed length of a sequence
        """
        l = self._length_rng.choice(self.all_lengths)
        return l

    def __len__(self) -> int:
        """Returns the length of this object"""
        return len(self.structures)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)

        angles = self.structures[index]["angles"]
        assert angles is not None
        # Pad or trim to given length
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
            logging.debug(f"Padded {orig_shape} -> {angles.shape}")
        elif angles.shape[0] > self.pad:
            angles = angles[: self.pad]

        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        retval = torch.from_numpy(angles).float()
        return {"angles": retval, "attn_mask": attn_mask, "position_ids": position_ids}


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
        dset_key: Optional[str] = None,
        timesteps: int = 1000,
        beta_schedule: beta_schedules.SCHEDULES = "linear",
        modulo: Optional[Union[float, Iterable[float]]] = None,
        noise_by_modulo: bool = False,
    ) -> None:
        super().__init__()
        self.dset = dset
        self.dset_key = dset_key

        self.modulo = modulo
        self.noise_by_modulo = noise_by_modulo

        self.timesteps = timesteps
        self.schedule = beta_schedule

        self.betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alphas = 1.0 - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def __len__(self) -> int:
        return len(self.dset)

    def sample_noise_adaptive(self, vals: torch.Tensor) -> torch.Tensor:
        """
        Adaptively sample noise based on modulo. This is necessary because
        if we have a modulo term, then the minimum value is 0 and generating
        noise with 0 mean results in a weird bimodal distribution
        """
        noise = torch.randn_like(vals)
        # If modulo not given, or if noise_by_modulo is False, then just return noise
        if self.modulo is None or not self.noise_by_modulo:
            return noise
        # Module is being used -- shift the noise
        assert self.modulo is not None
        try:
            centers = torch.tensor([m / 2 if m > 0 else 0 for m in self.modulo])
            v = torch.tensor([m / 6 if m > 0 else 1 for m in self.modulo])
            noise = noise * v + centers
        except TypeError:
            noise = noise * self.modulo / 6 + self.modulo / 2

        return noise

    def __getitem__(
        self, index, use_t_val: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Gets the i-th item in the dataset and adds noise
        use_t_val is useful for manually querying specific timepoints
        """
        item = self.dset.__getitem__(index)
        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key]
        else:
            vals = item
        assert isinstance(vals, torch.Tensor), f"Expected tensor but got {type(vals)}"

        # Sample a random timepoint and add corresponding noise
        if use_t_val is not None:
            t_val = np.clip(np.array([use_t_val]), 0, self.timesteps - 1)
            t = torch.from_numpy(t_val).long()
        else:
            t = torch.randint(0, self.timesteps, (1,)).long()
        sqrt_alphas_cumprod_t = utils.extract(self.sqrt_alphas_cumprod, t, vals.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, vals.shape
        )
        noise = self.sample_noise_adaptive(vals)
        noised_vals = (
            sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
        )

        # If modulo is given ensure that we do modulo
        if self.modulo is not None:
            try:
                assert (
                    len(self.modulo) == noised_vals.shape[1]
                ), f"Mismatched shapes: {noised_vals.shape} vs {len(self.modulo)} modulo terms"
                # Should have shape (seq_len, 4)
                noised_vals = torch.vstack(
                    [t % m if m != 0 else t for t, m in zip(noised_vals.T, self.modulo)]
                ).T
            except TypeError:
                # not iterable --> float
                noised_vals = noised_vals % self.modulo

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


class GaussianDistUniformAnglesNoisedAnglesDataset(NoisedAnglesDataset):
    """
    Same as NoisedAnglesDataset but with uniform noise for the angles. Importantly, we keep
    the Gaussian noise for the distances.
    """

    def sample_noise_adaptive(self, vals: torch.Tensor) -> torch.Tensor:
        """Sample Gaussian AND uniform noise for the dist and angles, respectively"""
        assert self.modulo is not None, "Must provide modulo for uniform noise"
        assert self.noise_by_modulo, "Must noise using modulo for uniform noise"

        assert (
            vals.shape[1] == 4
        ), f"Expected vals to have shape (seqlen, 4) but got {vals.shape}"
        g_noise = torch.randn((vals.shape[0], 1))
        uni_noise = torch.rand((vals.shape[0], vals.shape[1] - 1))
        noise = torch.cat([g_noise, uni_noise], dim=1)
        assert (
            noise.shape == vals.shape
        ), f"Expected noise to have shape {vals.shape} but got {noise.shape}"

        # Modulo is being used -- shift noise
        assert self.modulo is not None
        try:
            scales = torch.tensor([1 if m == 0 else m for m in self.modulo])
            logging.debug(f"Scaling by {scales}")
            # Scaling the first column, which is the Gaussian noise, by 1 doesn't matter
            assert scales[0] == 1.0
            noise = noise * scales
        except TypeError:
            noise = noise * self.modulo

        return noise


def coords_to_angles(
    coords: Dict[str, List[List[float]]], shift_angles_positive: bool = True
) -> Optional[np.ndarray]:
    """
    Sanitize the coordinates to not have NaN and convert them into
    arrays of angles. If sanitization fails, return None

    if shift_angles_positive, take the angles given in [-pi, pi] range and
    shift them to [0, 2pi] range
    """
    first_valid_idx, last_valid_idx = 0, len(coords["N"])

    # Walk through coordinates and trim trailing nan
    for k in ["N", "CA", "C"]:
        logging.debug(f"{k}:\t{coords[k][:5]}")
        arr = np.array(coords[k])
        # Get all valid indices
        valid_idx = np.where(~np.any(np.isnan(arr), axis=1))[0]
        first_valid_idx = max(first_valid_idx, np.min(valid_idx))
        last_valid_idx = min(last_valid_idx, np.max(valid_idx) + 1)
    logging.debug(f"Trimming nans keeps {first_valid_idx}:{last_valid_idx}")
    for k in ["N", "CA", "C"]:
        coords[k] = coords[k][first_valid_idx:last_valid_idx]
        arr = np.array(coords[k])
        if np.any(np.isnan(arr)):
            logging.debug("Got nan in middle of array")
            return None
    angles = pdb_utils.process_coords(coords)
    # https://www.rosettacommons.org/docs/latest/application_documentation/trRosetta/trRosetta#application-purpose_a-note-on-nomenclature
    # omega = inter-residue dihedral angle between CA/CB of first and CB/CA of second
    # theta = inter-residue dihedral angle between N, CA, CB of first and CB of second
    # phi   = inter-residue angle between CA and CB of first and CB of second
    dist, omega, theta, phi = angles
    assert dist.shape == omega.shape == theta.shape == phi.shape
    logging.debug(f"Pre slice shape: {dist.shape, omega.shape, theta.shape, phi.shape}")
    # Slice out so that we have the angles and distances between the n and n+1 items
    n = dist.shape[0]
    indices_i = np.arange(n - 1)
    indices_j = indices_i + 1
    dist_slice = dist[indices_i, indices_j]
    omega_slice = omega[indices_i, indices_j]
    theta_slice = theta[indices_i, indices_j]
    phi_slice = phi[indices_i, indices_j]
    logging.debug(
        f"Post slice shape: {dist_slice.shape, omega_slice.shape, theta_slice.shape, phi_slice.shape}"
    )
    all_values = np.array([dist_slice, omega_slice, theta_slice, phi_slice]).T
    assert all_values.shape == (n - 1, 4)

    assert np.all(
        np.logical_and(all_values[:, 1:] <= np.pi, all_values[:, 1:] >= -np.pi,)
    ), "Angle values outside of [-pi, pi] range"
    if shift_angles_positive:
        all_values[:, 1:] += np.pi

    return all_values


def main():
    dset = CathConsecutiveAnglesDataset(toy=True, split="train")
    noised_dset = GaussianDistUniformAnglesNoisedAnglesDataset(
        dset,
        dset_key="angles",
        modulo=[0, 2 * np.pi, 2 * np.pi, 2 * np.pi],
        noise_by_modulo=True,
    )
    x = noised_dset[0]
    # for k, v in x.items():
    #     print(k)
    #     print(v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
