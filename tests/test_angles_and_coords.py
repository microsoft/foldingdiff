import os
import tempfile
import unittest

import numpy as np
from biotite.structure.io.pdb import PDBFile

from foldingdiff import angles_and_coords as ac


def get_structure_coords(fname: str) -> np.ndarray:
    """ """
    with open(fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]
    return chain.coord


class TestAddingSidechains(unittest.TestCase):
    """
    Test that sidechains are "grafted" on corectly
    """

    def test_simple(self):
        fname = os.path.join(os.path.dirname(__file__), "../data/all_residues.pdb")
        backbone = ac.extract_backbone_coords(fname, ["N", "CA", "C"])
        with tempfile.TemporaryDirectory() as td:
            backbone_fname = os.path.join(td, "backbone_only.pdb")

            ac.write_coords_to_pdb(backbone, backbone_fname)
            ac.add_sidechains_to_backbone(
                backbone_fname,
                "RHKDESTNQCGPAVILMFYW",
                os.path.join(td, "out.pdb"),
                reference_pdbs=(fname,),
            )

            ref_coords = get_structure_coords(fname)
            test_coords = get_structure_coords(os.path.join(td, "out.pdb"))
            self.assertTrue(np.allclose(ref_coords, test_coords))
