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
from biotite.structure.io.pdbx import PDBxFile

import torch
from torch.utils.data import Dataset

import nerf


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
) -> Optional[pd.DataFrame]:
    """Parse the pdb file for the given values"""
    assert os.path.isfile(fname)
    source = PDBFile.read(str(fname))
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    # First get the dihedrals
    phi, psi, omega = struc.dihedral_backbone(source_struct)
    calc_angles = {"phi": phi, "psi": psi, "omega": omega}

    # Get any additional angles
    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    # Gets the N - CA - C for each residue
    # https://www.biotite-python.org/apidoc/biotite.structure.filter_backbone.html
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    for a in non_dihedral_angles:
        if a == "tau" or a == "N:CA:C":
            # tau = N - CA - C internal angles
            idx = np.array(
                [list(range(i, i + 3)) for i in range(3, len(backbone_atoms), 3)]
                + [(0, 0, 0)]
            )
        elif a == "CA:C:1N":  # Same as C-N angle in nerf
            # This measures an angle between two residues. Due to the way we build
            # proteins out later, we do not need to meas
            idx = np.array(
                [(i + 1, i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        elif a == "C:1N:1CA":
            idx = np.array(
                [(i + 2, i + 3, i + 4) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx)

    # At this point we've only looked at dihedral and angles; check value range
    for k, v in calc_angles.items():
        assert (
            np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi
        ), f"Illegal values for {k}"

    # Get any additional distances
    for d in distances:
        if (d == "0C:1N") or (d == "C:1N"):
            # Since this is measuring the distance between pairs of residues, there
            # is one fewer such measurement than the total number of residues like
            # for dihedrals. Therefore, we pad this with a null 0 value at the end.
            idx = np.array(
                [(i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0)]
            )
        elif d == "N:CA":
            # We start resconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally we pad with a
            # null value at the end
            idx = np.array(
                [(i, i + 1) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        elif d == "CA:C":
            # We start reconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally, we pad with a
            # null value at the end.
            idx = np.array(
                [(i + 1, i + 2) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        else:
            raise ValueError(f"Unrecognized distance: {d}")
        calc_angles[d] = struc.index_distance(backbone_atoms, indices=idx)

    return pd.DataFrame({k: calc_angles[k].squeeze() for k in distances + angles})


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


def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: List[str] = ["phi", "psi", "omega"],
    dists_to_set: List[str] = [],
    center_coords: bool = True,
) -> None:
    """Create a new chain using NERF to convert to cartesian coordinates"""
    # Check that we are at least setting the dihedrals
    required_dihedrals = ["phi", "psi", "omega"]
    assert all([a in angles_to_set for a in required_dihedrals])

    nerf_build_kwargs = dict(
        phi_dihedrals=dists_and_angles["phi"],
        psi_dihedrals=dists_and_angles["psi"],
        omega_dihedrals=dists_and_angles["omega"],
    )
    for a in angles_to_set:
        if a in required_dihedrals:
            continue
        assert a in dists_and_angles
        if a == "tau" or a == "N:CA:C":
            nerf_build_kwargs["bond_angle_ca_c"] = dists_and_angles[a]
        elif a == "CA:C:1N":
            nerf_build_kwargs["bond_angle_c_n"] = dists_and_angles[a]
        elif a == "C:1N:1CA":
            nerf_build_kwargs["bond_angle_n_ca"] = dists_and_angles[a]
        else:
            raise ValueError(f"Unrecognized angle: {a}")

    for d in dists_to_set:
        assert d in dists_and_angles.columns
        if d == "0C:1N":
            nerf_build_kwargs["bond_len_c_n"] = dists_and_angles[d]
        elif d == "N:CA":
            nerf_build_kwargs["bond_len_n_ca"] = dists_and_angles[d]
        elif d == "CA:C":
            nerf_build_kwargs["bond_len_ca_c"] = dists_and_angles[d]
        else:
            raise ValueError(f"Unrecognized distance: {d}")

    nerf_builder = nerf.NERFBuilder(**nerf_build_kwargs)
    coords = (
        nerf_builder.centered_cartesian_coords
        if center_coords
        else nerf_builder.cartesian_coords
    )

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 3),
        3,
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
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
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    # Add bonds
    full_structure.bonds = struc.BondList(full_structure.array_length())
    indices = list(range(full_structure.array_length()))
    for a, b in zip(indices[:-1], indices[1:]):
        full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

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

    create_new_chain_nerf(
        "test.pdb",
        vals,
        angles_to_set=["phi", "psi", "omega", "tau"],
        dists_to_set=["0C:1N"],
    )
    new_vals = canonical_distances_and_dihedrals("test.pdb")
    print(new_vals[:10])

    # score = tmalign.run_tmalign("test.pdb", reference_fname)
    # print(score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_reverse_dihedral()
    test_generation()
