"""
Script to add oxygen to backbone. This is not strictly included in the original
set of dihedral/bond angles, and it is not technically a part of the backbone,
but is required for some downstream tools to work (e.g. pymol). This script is
meant to be run directly on generated backbones, that have not yet have side
chains added onto them.

Example usage:
python add_oxygen_to_backbone sampled_pdb sampled_pdb_with_o
"""

import os
import logging
import argparse
import glob

from biotite import structure as struct
from biotite.structure.io.pdb import PDBFile

from tqdm.auto import tqdm

from foldingdiff import nerf

logging.basicConfig(level=logging.INFO)


def read_structure(fname: str) -> struct.AtomArray:
    """Return an atom array from the given pdb file."""
    with open(fname) as source:
        pdb_file = PDBFile.read(source)
    assert pdb_file.get_model_count() == 1
    structure = pdb_file.get_structure()[0]
    return structure


def add_oxygen_to_backbone(structure: struct.AtomArray) -> struct.AtomArray:
    """Returns a new atom array with oxygen atoms added to the backbone."""
    assert len(structure) % 3 == 0
    assert struct.get_residue_count(structure) == len(structure) // 3

    retval = []
    for i, atom in enumerate(structure):
        atom.atom_id = len(retval) + 1
        atom.res_id = i // 3
        retval.append(atom)
        # Last atom in residue after (0, N), (1, CA), (2, C)
        if i % 3 == 2 and i + 1 < len(structure):
            # Insert oxygen
            psi = struct.dihedral(
                structure[i - 2].coord,
                structure[i - 1].coord,
                structure[i].coord,
                structure[i + 1].coord,
            )
            oxy = struct.Atom(
                coord=nerf.place_dihedral(
                    retval[-3].coord,
                    retval[-2].coord,
                    retval[-1].coord,
                    torsion_angle=psi,
                    bond_angle=2.0992622,
                    bond_length=1.2359372,
                ),
                chain_id=retval[-1].chain_id,
                res_id=retval[-1].res_id,
                atom_id=len(retval) + 1,
                res_name=retval[-1].res_name,
                atom_name="O",
                element="O",
            )
            # Propogate any other annotations
            for k in retval[-1]._annot.keys():
                if k not in oxy._annot:
                    oxy._annot[k] = retval[-1]._annot[k]
            retval.append(oxy)
    return struct.array(retval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str, help="Input directory with .pdb files")
    parser.add_argument("outdir", type=str, help="Output directory to write .pdb files")
    args = parser.parse_args()

    pdb_files = list(glob.glob(os.path.join(args.indir, "*.pdb")))
    logging.info(f"Found {len(pdb_files)} pdb files in {args.indir}")

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    for fname in tqdm(pdb_files):
        structure = read_structure(fname)
        updated_backbone_with_o = add_oxygen_to_backbone(structure)
        outname = os.path.join(args.outdir, os.path.basename(fname))
        with open(outname, "w") as sink:
            pdb_file = PDBFile()
            pdb_file.set_structure(updated_backbone_with_o)
            pdb_file.write(sink)
        del pdb_file


if __name__ == "__main__":
    main()
