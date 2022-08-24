"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

import pickle
import functools
import multiprocessing
import os, sys
import glob
import gzip
import logging
import json
from typing import *
import torch


from Bio import PDB
from Bio.PDB import ic_rebuild

from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

CATH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cath"
)

ALPHAFOLD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/alphafold"
)

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

    Source for data splits:
    - https://www.mit.edu/~vgarg/GenerativeModelsForProteinDesign.pdf

    For all domains in the CATH 4.2 40% non-redundant set of proteins, we obtained full chains up to length
    500 and then randomly assigned their CATH topology classifications (CAT codes) to train, validation
    and test sets at a targeted 80/10/10 split. Since each chain can contain multiple CAT codes, we first
    removed any redundant entries from train and then from validation. Finally, we removed any chains
    from the test set that had CAT overlap with train and removed chains from the validation set with
    CAT overlap to train or test. This resulted in a dataset of 18024 chains in the training set, 608 chains
    in the validation set, and 1120 chains in the test set.
    """

    feature_names = {
        "angles": ["bond_dist", "omega", "theta", "phi"],
    }
    feature_is_angular = {
        "angles": [False, True, True, True],
    }

    def __init__(
        self,
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 512,
        shift_to_zero_twopi: bool = False,
        toy: Union[bool, int] = False,
    ) -> None:
        super().__init__()
        assert os.path.isdir(CATH_DIR), f"Expected CATH dir at {CATH_DIR}"
        # Determine limit on reading based on toy argument
        item_limit = None
        if toy is None:
            pass
        elif isinstance(toy, bool) and toy:
            item_limit = 150
        elif isinstance(toy, int):
            item_limit = toy
        else:
            raise ValueError(f"Unrecognized value for toy: {toy} (type {type(toy)})")

        self.pad = pad
        self.shift_to_zero_twopi = shift_to_zero_twopi
        # json list file -- each line is a json
        data_file = os.path.join(CATH_DIR, "chain_set.jsonl.gz")
        assert os.path.isfile(data_file)
        self.structures = []
        with gzip.open(data_file) as source:
            for i, line in enumerate(source):
                structure = json.loads(line.strip())
                self.structures.append(structure)
                if item_limit and i >= item_limit:
                    logging.warning(f"Truncating CATH to {item_limit} structures")
                    break

        # Get data split if given
        self.split = split
        if self.split is not None:
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
        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

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

    def __str__(self) -> str:
        return f"CathConsecutiveAnglesDataset {self.split} split with {len(self)} structures"

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
        assert sum(attn_mask) == l

        # Perform padding
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
        assert angles.shape == (self.pad, 4)

        # Shift to [0, 2pi] if configured as such
        if self.shift_to_zero_twopi:
            angles[:, 1:] += np.pi  # Shift from [-pi, pi] to [0, 2pi]
            assert angles[:, 1:].min() >= 0
            assert angles[:, 1:].max() < 2 * np.pi


        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        angles = torch.from_numpy(angles).float()

        retval = {
            "angles": angles,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
        }
        for k, v in self.feature_names.items():
            assert retval[k].shape == (self.pad, len(v))
        return retval


def read_and_extract_angles_from_pdb(
    fname: str, force_compute: bool = False, write_cache: bool = True
) -> Dict[str, Any]:
    """
    Helper function for reading and computing angles from pdb file
    """
    # Check if cached computed results exists
    # https://stackoverflow.com/questions/52444921/save-numpy-array-using-pickle
    suffix = fname.split(".")[-1]
    cached_fname = os.path.join(
        os.path.dirname(os.path.abspath(fname)),
        os.path.basename(fname).replace(suffix, "extracted.pkl"),
    )
    if os.path.isfile(cached_fname) and not force_compute:
        logging.debug(f"Loading cached values from {cached_fname}")
        with open(cached_fname, "rb") as f:
            return pickle.load(f)

    # Perform the computation
    atoms = ["N", "CA", "C"]
    coords, seq, valid_idx = pdb_utils.parse_PDB(fname, atoms=atoms)
    assert coords.shape[0] == len(
        seq
    ), f"Mismatched lengths: {coords.shape[0]} vs {len(seq)} in {fname}"
    # coords has shape (length, atoms, 3)
    coords_dict = {atom: coords[:, i, :] for i, atom in enumerate(atoms)}
    angles = coords_to_angles(coords_dict, shift_angles_positive=True)
    retval = {"coords": coords, "angles": angles, "seq": seq, "valid_idx": valid_idx}
    # Cache the result
    if write_cache:
        with open(cached_fname, "wb") as f:
            logging.debug(f"Dumping cached values from {fname} to {cached_fname}")
            pickle.dump(retval, f)

    return retval


class CathCanonicalAnglesDataset(Dataset):
    """
    Load in the dataset.

    All angles should be given between [-pi, pi]
    """

    feature_names = {"angles": ["bond_dist", "phi", "psi", "omega", "tau"]}
    feature_is_angular = {"angles": [False, True, True, True, True]}
    cache_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache_canonical_structures.pkl"
    )

    def __init__(
        self,
        split: Optional[Literal["train", "test", "validation"]] = None,
        pad: int = 512,
        toy: int = 0,
        shift_to_zero_twopi: bool = False,
    ) -> None:
        super().__init__()
        assert not shift_to_zero_twopi, "No reason to shift to zero twopi"
        self.pad = pad

        # gather files
        fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
        assert fnames, f"No files found in {CATH_DIR}/dompdb"

        # self.structures should be a list of dicts
        # Always compoute for toy; do not save
        if toy:
            if isinstance(toy, bool):
                toy = 150
            fnames = fnames[:toy]

            logging.info(f"Loading toy dataset of {toy} structures")
            struct_arrays = [canonical_angles_from_fname(f) for f in fnames]
            self.structures = []
            for fname, s in zip(fnames, struct_arrays):
                if s is None:
                    continue
                self.structures.append({"angles": s, "fname": fname})
        elif not os.path.exists(self.cache_fname):
            logging.info(
                f"Computing full dataset of {len(fnames)} with {multiprocessing.cpu_count()} threads"
            )
            # Generate dihedral angles
            # https://biopython.org/docs/1.76/api/Bio.PDB.PDBParser.html
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            struct_arrays = pool.map(canonical_angles_from_fname, fnames, chunksize=250)
            pool.close()
            pool.join()

            # Contains only non-null structures
            self.structures = []
            for fname, s in zip(fnames, struct_arrays):
                if s is None:
                    continue
                self.structures.append({"angles": s, "fname": fname})
            # Write the output to a file for faster loading next time
            logging.info(f"Saving full dataset to {self.cache_fname}")
            with open(self.cache_fname, "wb") as sink:
                pickle.dump(self.structures, sink)
        else:
            logging.info(f"Loading cached full dataset from {self.cache_fname}")
            with open(self.cache_fname, "rb") as source:
                self.structures = pickle.load(source)

        # Split the dataset if requested. This is implemented here to maintain
        # functional parity with the original CATH dataset. Original CATH uses
        # a 80/10/10 split
        rng = np.random.default_rng(seed=6489)
        # Shuffle the sequences so contiguous splits acts like random splits
        rng.shuffle(self.structures)
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

        # Aggregate lengths
        self.all_lengths = [s["angles"].shape[0] for s in self.structures]
        self._length_rng = np.random.default_rng(seed=6489)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

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

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        angles = self.structures[index]["angles"]
        assert angles is not None

        # Pad/trim and create attention mask. 0 indicates masked
        l = min(self.pad, angles.shape[0])
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:l] = 1.0

        # Additionally, mask out positions that are nan
        # is_nan = np.where(np.any(np.isnan(angles), axis=1))[0]
        # attn_mask[is_nan] = 0.0  # Mask out the nan positions

        if angles.shape[0] < self.pad:
            angles = np.pad(
                angles,
                ((0, self.pad - angles.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif angles.shape[0] > self.pad:
            angles = angles[: self.pad]
        assert angles.shape == (self.pad, len(self.feature_names["angles"]))

        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)
        angles = torch.from_numpy(angles).float()

        retval = {
            "angles": angles,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
        }
        return retval


def canonical_angles_from_fname(
    fname: str,
    angles=["phi", "psi", "omega", "tau"],
    distances=["0C:1N"],
    use_radians: bool = True,
) -> Optional[np.ndarray]:
    """
    Parse PDB from fname. Returns an array of distance and angles
    https://foldit.fandom.com/wiki/Backbone_angle - There are

    https://biopython.org/wiki/Reading_large_PDB_files
    """
    parser = PDB.PDBParser(QUIET=True)

    s = parser.get_structure("", fname)
    # s.atom_to_internal_coordinates()
    # s.internal_to_atom_coordinates()

    # If there are multiple chains then skip and return None
    chains = [c for c in s.get_chains()]
    if len(chains) > 1:
        logging.warning(f"{fname} has multiple chains, returning None")
        return None
    chain = chains[0]
    chain.atom_to_internal_coordinates()

    residues = [r for r in chain.get_residues()]

    values = []
    # https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html#Bio.PDB.internal_coords.IC_Chain
    ic = chain.internal_coord  # Type IC_Chain
    if not ic_rebuild.structure_rebuild_test(chain)["pass"]:
        # https://biopython.org/docs/dev/api/Bio.PDB.ic_rebuild.html#Bio.PDB.ic_rebuild.structure_rebuild_test
        logging.warning(f"{fname} failed rebuild test, returning None")
        return None

    # Attributes
    # - dAtoms: homogeneous atom coordinates (4x4) of dihedra, second atom at origin
    # - hAtoms: homogeneous atom coordinates (3x4) of hedra, central atom at origin
    # - dihedra: Dihedra forming residues in this chain; indexed by 4-tuples of AtomKeys.
    # - ordered_aa_ic_list: IC_Residue objects in order of appearance in the chain.
    # https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html#Bio.PDB.internal_coords.IC_Residue
    for ric in ic.ordered_aa_ic_list:
        # https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html#Bio.PDB.internal_coords.IC_Residue.pick_angle
        this_dists = np.array([ric.get_length(d) for d in distances], dtype=np.float64)
        this_angles = np.array([ric.get_angle(a) for a in angles], dtype=np.float64)
        this_angles_nonnan = ~np.isnan(this_angles)
        if use_radians:
            this_angles = this_angles / 180 * np.pi
            assert np.all(this_angles[this_angles_nonnan] >= -np.pi) and np.all(
                this_angles[this_angles_nonnan] <= np.pi
            )
        else:
            assert np.all(this_angles[this_angles_nonnan] >= -180) and np.all(
                this_angles[this_angles_nonnan] <= 180
            )
        values.append(np.concatenate((this_dists, this_angles)))

    retval = np.array(values, dtype=np.float64)
    np.nan_to_num(retval, copy=False)  # Replace nan with 0 and info with large num
    assert retval.shape == (len(residues), len(distances) + len(angles))
    return retval


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
        assert os.path.isdir(
            ALPHAFOLD_DIR
        ), f"Expected AlphaFold data dir at {ALPHAFOLD_DIR}"
        self.pad = pad
        self.shift_to_zero_twpi = shift_to_zero_twopi

        # Glob for the untarred files
        pdb_files = glob.glob(os.path.join(ALPHAFOLD_DIR, "*.pdb.gz"))
        pfunc = functools.partial(
            read_and_extract_angles_from_pdb, force_compute=force_recompute_angles
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
        dset_key: str,
        timesteps: int = 250,
        exhaustive_t: bool = False,
        beta_schedule: beta_schedules.SCHEDULES = "linear",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
        shift_to_zero_twopi: bool = False,
    ) -> None:
        super().__init__()
        self.dset = dset
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        assert (
            dset_key in dset.feature_is_angular
        ), f"{dset_key} not in {dset.feature_is_angular}"

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.shift_to_zero_twopi = shift_to_zero_twopi
        if hasattr(self.dset, "shift_angles_zero_twopi"):
            assert (
                self.shift_to_zero_twopi == self.dset.shift_to_zero_twopi
            ), "Mismatched shift_to_zero_twopi"
            logging.info(
                f"Checked shift_to_zero_twopi between noiser and underlying dset -- both {self.shift_to_zero_twopi}"
            )

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = beta_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)

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

        # Scale by provided variance scales based on angular or not
        if self.angular_var_scale != 1.0 or self.nonangular_var_scale != 1.0:
            for j in range(noise.shape[1]):
                s = (
                    self.angular_var_scale
                    if self.dset.feature_is_angular[self.dset_key][j]
                    else self.nonangular_var_scale
                )
                noise[:, j] *= s

        # Make sure that the noise doesn't run over the boundaries
        angular_idx = np.where(self.dset.feature_is_angular[self.dset_key])[0]
        noise[:, angular_idx] = utils.modulo_with_wrapped_range(
            noise[:, angular_idx], -np.pi, np.pi
        )
        if self.shift_to_zero_twopi:
            # Add pi
            noise[:, angular_idx] += np.pi

        return noise

    def __getitem__(
        self, index: int, use_t_val: Optional[int] = None
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
            item = self.dset.__getitem__(item_index)
        else:
            item = self.dset.__getitem__(index)

        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key]
        else:
            vals = item
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
        if self.shift_to_zero_twopi:
            noised_vals[:, angular_idx] = utils.modulo_with_wrapped_range(
                noised_vals[:, angular_idx], 0, 2 * np.pi
            )
        else:
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


class GaussianDistUniformAnglesNoisedDataset(NoisedAnglesDataset):
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


def coords_to_angles(
    coords: Union[np.ndarray, Dict[str, List[List[float]]]],
) -> Optional[np.ndarray]:
    """
    Sanitize the coordinates to not have NaN and convert them into
    arrays of angles. If sanitization fails, return None

    Returned angles given in [-pi, pi] range
    """
    if isinstance(coords, dict):
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
    ), "Angle values outside of expected [-pi, pi] range"
    return all_values


def main():
    # dset = AlphafoldConsecutiveAnglesDataset(force_recompute_angles=False, toy=False)
    # print(dset)
    # dset = CathConsecutiveAnglesDataset(toy=10, split="train")
    # noised_dset = SynNoisedMaskedOnlyDataset(dset, dset_key="angles",)
    # print(len(noised_dset))
    # print(noised_dset[0])

    dset = CathCanonicalAnglesDataset()
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
