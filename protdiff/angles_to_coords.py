"""
Code to convert from angles between residues to XYZ coordinates. 

Based on: 
https://github.com/biopython/biopython/blob/master/Bio/PDB/ic_rebuild.py
"""
import os
from typing import *

import numpy as np
import scipy.linalg

from Bio import PDB
from Bio.PDB import PICIO


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


def create_new_chain(out_fname: str = "temp.pdb", n: int = 5):
    """
    Creates a new chain

    USeful references:
    https://stackoverflow.com/questions/47631064/create-a-polymer-chain-of-nonstandard-residues-from-a-single-residue-pdb
    """
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
    aa_ic = []
    for resnum, aa in enumerate("A" * n):  # Alanine is a single carbon sidechain
        # Constructor is ID, resname, segID
        # ID is 3-tuple of example (' ', 85, ' ')
        # resnum uses 1-indexing in real PDB files
        res = PDB.Residue.Residue((" ", resnum + 1, " "), "ALA", "A")
        # atoms in each resiude are N, CA, C, O, CB
        for atom in ["N", "CA", "C", "O", "CB"]:
            # https://biopython.org/docs/1.76/api/Bio.PDB.Atom.html
            # constructor expects
            # name, coord, bfactor, occupancy, altloc, fullname, serial_number
            # Generate a random coordinate
            # Occupancy is typically 1.0
            # Values under 10 create a model of the atom that is very sharp, indicating that the atom is not moving much and is in the same position in all of the molecules in the crystal
            # Values greater than 50 or so indicate that the atom is moving so much that it can barely been seen.
            atom_obj = PDB.Atom.Atom(
                atom, rng.random(3), 10.0, 1.0, " ", atom, resnum, element=atom[:1]
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

    # Create placeholder values
    ic.atom_to_internal_coordinates()
    # ic.set_residues()
    for ric in ic.ordered_aa_ic_list:
        assert isinstance(ric, PDB.internal_coords.IC_Residue)
        # Random values for now
        ric.set_angle("phi", 0.5)
        ric.set_angle("psi", 1.0)
        ric.set_angle("omega", -1.0)
        ric.set_angle("tau", -1.5)
        ric.set_length("0C:1N", 1.1)
    chain.internal_coord = ic

    chain.internal_to_atom_coordinates()

    # Write output
    io = PDB.PDBIO()
    io.set_structure(chain)
    io.save("final.pdb")


def reverse_dihedral(v1, v2, v3, dihedral):
    """
    Find vector from c->d given a, b, c, & dihedral angle formed by a, b, c, d
    """
    # see https://github.com/pycogent/pycogent/blob/master/cogent/struct/dihedral.py
    def rotate(v, theta, axis):
        # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
        m = scipy.linalg.expm(
            np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta)
        )
        return np.dot(v, m)

    v12 = v2 - v1
    v23 = v3 - v2
    # This is the first vector in the angle calculation that gives dihedral
    normal1 = np.cross(v12, v23)
    normal1 = normal1 / scipy.linalg.norm(normal1)

    rotated = rotate(normal1, dihedral, v12)

    # Invert cross product
    # https://math.stackexchange.com/questions/32600/whats-the-opposite-of-a-cross-product
    num = np.cross(rotated, v23)
    den = np.dot(v23, v23.T)
    final_offset = num / den  # Corresponds to V34
    final_offset /= scipy.linalg.norm(final_offset)

    return final_offset


def test_generation():
    """
    Test the generation of a new chain
    """
    pdb_to_pic(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/7PFL.pdb"),
        "7PFL.pic",
    )
    pic_to_pdb("7PFL.pic", "7PFL.pdb")
    create_new_chain()


def test_reverse_dihedral():
    """
    Test that we can reverse a dihedral
    """
    from sequence_models import pdb_utils

    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    d = np.array([[-1.0, 1.0, 0.0]])
    dh = pdb_utils.get_dihedrals(a, b, c, d)
    print(dh)

    reverse_dihedral(a, b, c, dh)


if __name__ == "__main__":
    # test_reverse_dihedral()
    test_generation()
