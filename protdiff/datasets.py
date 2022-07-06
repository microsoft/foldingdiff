"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

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


class CathConsecutiveAnglesDataset(Dataset):
    """
    Represent proteins as their constituent angles instead of 3D coordinates

    The three angles phi, psi, and omega determine the backbone structure.
    Omega is typically fixed ~180 degrees in most cases.

    Useful reading:
    - https://proteinstructures.com/structure/ramachandran-plot/
    - https://foldit.fandom.com/wiki/Backbone_angle
    - http://www.cryst.bbk.ac.uk/PPS95/course/9_quaternary/3_geometry/torsion.html
    - https://swissmodel.expasy.org/course/text/chapter1.htm
    - https://www.nature.com/articles/s41598-020-76317-6
    - https://userguide.mdanalysis.org/1.1.1/examples/analysis/structure/dihedrals.html
    """

    def __init__(self, pad: int = 512, toy: bool = False) -> None:
        super().__init__()

        self.pad = pad
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

        # Generate angles in parallel and attach them to dictionary
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        angles = pool.map(
            coords_to_angles, [d["coords"] for d in self.structures], chunksize=250
        )
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
        all_lengths = [s["angles"].shape[1] for s in self.structures]
        logging.info(
            f"Length of angles: {np.min(all_lengths)}-{np.max(all_lengths)}, mean {np.mean(all_lengths)}"
        )

    def __len__(self) -> int:
        """Returns the length of this object"""
        return len(self.structures)

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)

        angles = self.structures[index]["angles"]
        assert angles is not None
        # Pad to given length
        if angles.shape[0] < self.pad:
            orig_shape = angles.shape
            angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            logging.debug(f"Padded {orig_shape} -> {angles.shape}")
        retval = torch.from_numpy(angles)
        return retval


class NoisedAnglesDataset(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__
    """

    def __init__(self, dset: Dataset, timesteps: int = 1000) -> None:
        super().__init__()
        self.dset = dset

        self.timesteps = timesteps
        self.betas = beta_schedules.linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        vals = self.dset.__getitem__(index)

        # Sample a random timepoint
        t = torch.randint(0, self.timesteps, (1,)).long()
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, vals.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, vals.shape
        )
        noise = torch.randn_like(vals)
        noised_vals = (
            sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return {
            "corrupted": noised_vals,
            "t": t,
            "known_noise": noise,
        }


def extract(a, t, x_shape):
    """
    Return the t-th item in a for each item in t
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def coords_to_angles(coords: Dict[str, List[List[float]]]) -> Optional[np.ndarray]:
    """
    Sanitize the coordinates to not have NaN and convert them into
    arrays of angles. If sanitization fails, return None
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
    return all_values


def main():
    dset = CathConsecutiveAnglesDataset()
    noised_dset = NoisedAnglesDataset(dset)
    x = noised_dset[0]
    for k, v in x.items():
        print(k)
        print(v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
