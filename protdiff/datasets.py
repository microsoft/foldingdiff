"""
Contains source code for loading in data and creating requisite PyTorch
data loader object
"""

import os, sys
import logging
import json

from tqdm.auto import tqdm

import numpy as np
from torch.utils.data import Dataset

CATH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cath"
)
assert os.path.isdir(CATH_DIR), f"Expected cath data at {CATH_DIR}"

from sequence_models import pdb_utils


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
        # https://www.rosettacommons.org/docs/latest/application_documentation/trRosetta/trRosetta#application-purpose_a-note-on-nomenclature
        # omega = inter-residue dihedral angle between CA/CB of first and CB/CA of second
        # theta = inter-residue dihedral angle between N, CA, CB of first and CB of second
        # phi   = inter-residue angle between CA and CB of first and CB of second
        dist, omega, theta, phi = angles
        logging.debug(
            f"Pre slice shape: {dist.shape, omega.shape, theta.shape, phi.shape}"
        )
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
        return dist_slice, omega_slice, theta_slice, phi_slice


def main():
    dset = CathConsecutiveAnglesDataset()
    print(dset[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
