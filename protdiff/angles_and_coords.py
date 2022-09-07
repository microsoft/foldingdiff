"""
Code to convert from angles between residues to XYZ coordinates. 

Based on: 
https://github.com/biopython/biopython/blob/master/Bio/PDB/ic_rebuild.py
"""
import os
import logging
import pickle
from typing import *

import numpy as np
import pandas as pd

from Bio import PDB
from Bio.PDB import PICIO, ic_rebuild
from sequence_models import pdb_utils

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

import torch
from torch.utils.data import Dataset

import pnerf  # Pytorch based
import nerf  # from medium
import mynerf  # self implementation


def pdb_to_pic(pdb_file: str, pic_file: str):
    """
    Convert the PDB file to a PIC file
    """
    parser = PDB.PDBParser(QUIET=True)
    s = parser.get_structure("pdb", pdb_file)
    chains = [c for c in s.get_chains()]
    if len(chains) > 1:
        raise NotImplementedError
    chain = chains.pop()  # type Bio.PDB.Chain.Chain
    # print(chain.__dict__.keys())

    # Convert to relative angles
    # Calculate dihedrals, angles, bond lengths (internal coordinates) for Atom data
    # Generates atomArray through init_edra
    chain.atom_to_internal_coordinates()

    for res in chain.internal_coord.ordered_aa_ic_list:
        # Look at only analines because that's what we generate
        if res.residue.get_resname() != "ALA":
            continue
        # print("REF", res, type(res))
        # print(res.dihedra.keys())

    with open(pic_file, "w") as sink:
        PICIO.write_PIC(chain, sink)


def pic_to_pdb(pic_file: str, pdb_file: str):
    """
    Read int he PIC file and convert to a PDB file
    """
    with open(pic_file) as source:
        f = PICIO.read_PIC(source)
    f.internal_to_atom_coordinates()

    io = PDB.PDBIO()
    io.set_structure(f)
    io.save(pdb_file)


def coords_to_trrosetta_angles(
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
        np.logical_and(
            all_values[:, 1:] <= np.pi,
            all_values[:, 1:] >= -np.pi,
        )
    ), "Angle values outside of expected [-pi, pi] range"
    return all_values


def trrosetta_angles_from_pdb(
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
    angles = coords_to_trrosetta_angles(coords_dict, shift_angles_positive=True)
    retval = {"coords": coords, "angles": angles, "seq": seq, "valid_idx": valid_idx}
    # Cache the result
    if write_cache:
        with open(cached_fname, "wb") as f:
            logging.debug(f"Dumping cached values from {fname} to {cached_fname}")
            pickle.dump(retval, f)

    return retval


def canonical_distances_and_dihedrals(
    fname: str,
    distances: List[str] = ["0C:1N"],
    angles: List[str] = ["phi", "psi", "omega", "tau"],
    use_radians: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Parse PDB from fname. Returns an array of distance and angles
    https://foldit.fandom.com/wiki/Backbone_angle - There are

    https://biopython.org/wiki/Reading_large_PDB_files
    """
    parser = PDB.PDBParser(QUIET=True)

    s = parser.get_structure("", fname)

    # If there are multiple chains then skip and return None
    chains = [c for c in s.get_chains()]
    if len(chains) > 1:
        logging.warning(f"{fname} has multiple chains, returning None")
        return None
    chain = chains.pop()
    chain.atom_to_internal_coordinates()

    residues = [r for r in chain.get_residues() if r.get_resname() not in ("HOH", "NA")]

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
    assert retval.shape == (
        len(residues),
        len(distances) + len(angles),
    ), f"Got mismatched shapes {retval.shape} != {(len(residues), len(distances) + len(angles))}"
    return pd.DataFrame(retval, columns=distances + angles)


def sample_coords(
    fname: str,
    subset_residues: Optional[Collection[str]] = None,
    query_atoms: List[str] = ["N", "CA", "C", "O", "CB"],
) -> List[pd.DataFrame]:
    """
    Sample the atomic coordinates of Alanine atoms. Return a list of dataframes each containing these
    coordinates.

    We use this to help figure out where to initialize atoms when creating a new chain
    """
    atomic_coords = []

    parser = PDB.PDBParser(QUIET=True)
    s = parser.get_structure("", fname)
    for chain in s.get_chains():
        residues = [
            r for r in chain.get_residues() if r.get_resname() not in ("HOH", "NA")
        ]

        for res in residues:
            if subset_residues is not None and res.get_resname() not in subset_residues:
                continue
            coords = {}
            for atom in res.get_atoms():
                coords[atom.get_name()] = atom.get_coord()
            all_atoms_present = True

            for atom in query_atoms:
                if atom not in coords:
                    logging.debug(f"{atom} not found in {res.get_resname()}")
                    all_atoms_present = False
                    break

            if all_atoms_present:
                atomic_coords.append(
                    pd.DataFrame([coords[k] for k in query_atoms], index=query_atoms)
                )
    return atomic_coords


def create_new_chain(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: List[str] = ["phi", "psi", "omega", "tau"],
    distances_to_set: List[str] = ["0C:1N"],
    sampled_values_dset: Optional[Dataset] = None,
):
    """
    Creates a new chain. Note that input is radians and must be converted to normal degrees
    for PDB compatibility. If given, sampled_values_dset is used to sample values that are
    not provided in the given dists_and_angles.

    USeful references:
    https://stackoverflow.com/questions/47631064/create-a-polymer-chain-of-nonstandard-residues-from-a-single-residue-pdb
    """

    def map_colname(colname: str) -> str:
        """Map human readable names to more accurate names"""
        mapping_dict = {"bond_dist": "0C:1N"}
        return mapping_dict.get(colname, colname)

    n = len(dists_and_angles)
    dists_and_angles = dists_and_angles.copy()
    dists_and_angles.columns = [map_colname(c) for c in dists_and_angles.columns]
    logging.info(
        f"Creating new chain of {n} residues with input values {dists_and_angles.columns.tolist()} setting {angles_to_set + distances_to_set}"
    )
    chain = PDB.Chain.Chain("A")
    # Avoid nonetype error
    chain.parent = PDB.Structure.Structure("pdb")

    rng = np.random.default_rng(seed=6489)

    # Assembly code
    # https://github.com/biopython/biopython/blob/4765a829258a776ac4c03b20b509e2096befba9d/Bio/PDB/internal_coords.py#L1393
    # appears to depend on chain's ordered_aa_ic_list whcih is a list of IC_Residues
    # https://biopython.org/docs/latest/api/Bio.PDB.internal_coords.html?highlight=ic_chain#Bio.PDB.internal_coords.IC_Residue
    # IC_residue extends https://biopython.org/docs/1.76/api/Bio.PDB.Residue.html
    # Set these IC_Residues
    for resnum, aa in enumerate(["ALA"] * n):  # Alanine is a single carbon sidechain
        # Constructor is ID, resname, segID
        # ID is 3-tuple of example (' ', 85, ' ')
        # resnum uses 1-indexing in real PDB files
        res = PDB.Residue.Residue((" ", resnum + 1, " "), aa, "A")
        # select a coordinate template for this atom
        # atoms in each resiude are N, CA, C, O, CB
        for atom in ["N", "CA", "C", "O", "CB"]:
            # https://biopython.org/docs/1.76/api/Bio.PDB.Atom.html
            # constructor expects
            # name, coord, bfactor, occupancy, altloc, fullname, serial_number
            # Generate a random coordinate
            # Occupancy is typically 1.0
            # Values under 10 create a model of the atom that is very sharp, indicating that the atom is not moving much and is in the same position in all of the molecules in the crystal
            # Values greater than 50 or so indicate that the atom is moving so much that it can barely been seen.
            coord = rng.random(3)
            atom_obj = PDB.Atom.Atom(
                atom, coord, 10.0, 1.0, " ", atom, resnum, element=atom[:1]
            )
            res.add(atom_obj)
        chain.add(res)

        # Convert residue to ic_residue
        ic_res = PDB.internal_coords.IC_Residue(res)
        ic_res.gly_Cbeta = True
        assert ic_res.is20AA

    # Write an intermediate to make sure we are modifying
    # io = PDB.PDBIO()
    # io.set_structure(chain)
    # io.save("intermediate.pdb")

    # Finished setting up the chain, now get the internal coordinates
    ic = PDB.internal_coords.IC_Chain(chain)
    # Initialize internal_coord data for loaded Residues.
    # Add IC_Residue as .internal_coord attribute for each Residue in parent Chain;
    # populate ordered_aa_ic_list with IC_Residue references for residues which can be built (amino acids and some hetatms)
    # set rprev and rnext on each sequential IC_Residue
    # populate initNCaC at start and after chain breaks

    # Determine which of the values are angles and which are distances
    # angle_colnames = [c for c in dists_and_angles.columns if not ":" in c]
    # dist_colnames = [c for c in dists_and_angles.columns if ":" in c]

    def sample_val_from_dset(val_name: str) -> float:
        """Sample a value from the dataset"""
        idx = rng.integers(0, len(sampled_values_dset))
        # Choose a random item
        rand_item = sampled_values_dset[idx]
        # Choose a random value from the sequence
        a = rng.integers(0, torch.sum(rand_item["attn_mask"]).item())
        b = sampled_values_dset.feature_names["angles"].index(angle)
        v = rand_item["angles"][a, b]
        return v

    # Create placeholder values
    ic.atom_to_internal_coordinates()
    # ic.set_residues()
    for i, ric in enumerate(ic.ordered_aa_ic_list):
        assert isinstance(ric, PDB.internal_coords.IC_Residue)
        for angle in angles_to_set:
            # Angles are given in radians, convert them back to degrees
            if angle in dists_and_angles:
                v = dists_and_angles.iloc[i][angle]
            elif (
                sampled_values_dset is not None
                and angle in sampled_values_dset.feature_names["angles"]
            ):
                v = sample_val_from_dset(angle)
            else:
                raise ValueError
            assert -np.pi <= v <= np.pi, f"{angle} is out of range with value {v}"
            ric.set_angle(angle, v / np.pi * 180)

        for dist in distances_to_set:
            if dist in dists_and_angles.columns:
                d = dists_and_angles.iloc[i][dist]
            elif (
                sampled_values_dset is not None
                and dist in sampled_values_dset.feature_names["angles"]
            ):
                d = sample_val_from_dset(dist)
            else:
                raise ValueError

            if np.isclose(d, 0):
                continue

            ric.set_length(dist, d)

    chain.internal_coord = ic

    # Recalculate the atom coordinates
    chain.internal_to_atom_coordinates()

    # Write output
    io = PDB.PDBIO()
    io.set_structure(chain)
    io.save(out_fname)


def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    backend: str = "nerf",
):
    """Create a new chain using NERF to convert to cartesian coordinates"""
    angles_to_set = ["phi", "psi", "omega"]
    assert all([a in dists_and_angles.columns for a in angles_to_set])
    dihedral_values = dists_and_angles[angles_to_set].values

    if backend == "pnerf":
        arr_input = torch.from_numpy(dihedral_values).type(torch.float)
        assert torch.all(arr_input >= -torch.pi)
        assert torch.all(arr_input <= torch.pi)
        assert arr_input.shape == (dists_and_angles.shape[0], len(angles_to_set))
        points = pnerf.dihedral_to_point(arr_input.unsqueeze(1))
        assert points.shape[1] == 1
        coords = (
            pnerf.point_to_coordinate(points, num_fragments=None)
            .squeeze()
            .cpu()
            .numpy()
        )

    elif backend == "nerf":
        nerf_builder = nerf.NeRF()
        coords = nerf_builder.compute_positions(dihedral_values.flatten())

    elif backend == "mynerf":
        nerf_builder = mynerf.NERFBuilder(
            phi_dihedrals=dists_and_angles["phi"],
            psi_dihedrals=dists_and_angles["psi"],
            omega_dihedrals=dists_and_angles["omega"],
        )
        coords = nerf_builder.cartesian_coords

    else:
        raise ValueError(f"Unknown backend: {backend}")
    # assert coords.shape == (
    #     int(dists_and_angles.shape[0] * 3),
    #     3,
    # ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    atoms = []
    for i, (n_coord, ca_coord, c_coord) in enumerate(
        (coords[j : j + 3] for j in range(0, len(coords), 3))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)


def test_generation(
    reference_fname: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb"
    )
):
    """
    Test the generation of a new chain
    """
    import tmalign

    test = PDBFile.read(reference_fname)
    print(test.get_structure())

    vals = canonical_distances_and_dihedrals(reference_fname)
    print(vals.iloc[:10])

    create_new_chain_nerf("test.pdb", vals, backend="mynerf")
    new_vals = canonical_distances_and_dihedrals("test.pdb")
    print(new_vals[:10])

    score = tmalign.run_tmalign("test.pdb", reference_fname)
    print(score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_reverse_dihedral()
    test_generation()
