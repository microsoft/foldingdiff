"""
Code to convert from angles between residues to XYZ coordinates. 
"""
import functools
import gzip
import os
import logging
import pickle
from typing import *
import warnings

import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

from foldingdiff import nerf

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]

MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []


def canonical_distances_and_dihedrals(
    fname: str,
    distances: List[str] = MINIMAL_DISTS,
    angles: List[str] = MINIMAL_ANGLES,
) -> Optional[pd.DataFrame]:
    """Parse the pdb file for the given values"""
    assert os.path.isfile(fname)
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true_div.*")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    # First get the dihedrals
    try:
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
    except struc.BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None

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
        if not (np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi):
            logging.warning(f"Illegal values for {k} in {fname} -- skipping")
            return None

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


def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: Optional[List[str]] = None,
    dists_to_set: Optional[List[str]] = None,
    center_coords: bool = True,
) -> str:
    """
    Create a new chain using NERF to convert to cartesian coordinates. Returns
    the path to the newly create file if successful, empty string if fails.
    """
    if angles_to_set is None and dists_to_set is None:
        angles_to_set, dists_to_set = [], []
        for c in dists_and_angles.columns:
            # Distances are always specified using one : separating two atoms
            # Angles are defined either as : separating 3+ atoms, or as names
            if c.count(":") == 1:
                dists_to_set.append(c)
            else:
                angles_to_set.append(c)
        logging.debug(f"Auto-determined setting {dists_to_set, angles_to_set}")
    else:
        assert angles_to_set is not None
        assert dists_to_set is not None

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
    if np.any(np.isnan(coords)):
        logging.warning(f"Found NaN values, not writing pdb file {out_fname}")
        return ""

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 3),
        3,
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    return write_coords_to_pdb(coords, out_fname)


def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 3 == 0
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
    return out_fname


@functools.lru_cache(maxsize=8192)
def get_pdb_length(fname: str) -> int:
    """
    Get the length of the chain described in the PDB file
    """
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    structure = PDBFile.read(fname)
    if structure.get_model_count() > 1:
        return -1
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    l = int(len(backbone) / 3)
    return l


def extract_backbone_coords(
    fname: str, atoms: Collection[Literal["N", "CA", "C"]] = ["CA"]
) -> Optional[np.ndarray]:
    """Extract the coordinates of the alpha carbons"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        return None
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    ca = [c for c in backbone if c.atom_name in atoms]
    coords = np.vstack([c.coord for c in ca])
    return coords


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_reverse_dihedral()
    print(
        extract_backbone_coords(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
        )
    )
