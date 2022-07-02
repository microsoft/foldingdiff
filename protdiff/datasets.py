"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

import os, sys
import logging
import json

from tqdm.auto import tqdm

from torch.utils.data import Dataset

CATH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cath"
)
assert os.path.isdir(CATH_DIR), f"Expected cath data at {CATH_DIR}"

from sequence_models import pdb_utils


class CathAnglesDataset(Dataset):
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

    def __init__(self) -> None:
        super().__init__()

        # json list file -- each line is a json
        data_file = os.path.join(CATH_DIR, "chain_set.jsonl")
        assert os.path.isfile(data_file)
        self.structures = []
        with open(data_file) as source:
            for i, line in enumerate(source):
                structure = json.loads(line.strip())
                assert (
                    len(structure["seq"])
                    == len(structure["coords"]["N"])
                    == len(structure["coords"]["CA"])
                    == len(structure["coords"]["C"])
                ), f"Unmatched sequence lengths at line {i}"
                self.structures.append(structure)

    def __len__(self) -> int:
        """Returns the length of this object"""
        return len(self.structures)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        structure = self.structures[index]
        angles = pdb_utils.process_coords(structure["coords"])
        dist, omega, theta, phi = angles


def main():
    dset = CathAnglesDataset()
    print(dset[1])


if __name__ == "__main__":
    main()
