"""
Script to identify and count van der waals clashes in a PDB file.

VDW values are taken from the following:
- https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
- https://pubs.acs.org/doi/epdf/10.1021/j100785a001

Useful references:
- https://www.pnas.org/doi/10.1073/pnas.072665799

Usage:

"""
import warnings
from typing import Collection, Dict
import multiprocessing as mp

import numpy as np

from tqdm.auto import tqdm

from scipy.spatial.distance import pdist, squareform

from biotite.structure.io.pdb import PDBFile
from biotite import structure as struct

# Van der waals in Angstroms
VDW_RADII = {
    "C": 1.7,
    "N": 1.55,
}


def count_clashes(fname: str, alpha: float = 0.63) -> int:
    """Counts the number of clashes in a PDB file."""

    # Read in the PDB file
    pdb_file = PDBFile.read(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = pdb_file.get_structure()[0]
    atoms = atoms[struct.filter_backbone(atoms)]

    # Compute pairwise distances
    pairwise_distances = squareform(pdist(atoms.coord))

    # Calculate the clash distance for each pair of atoms
    # Default value is 0 to indicate that the pairwise distance cannot clash
    clash_distance = np.zeros_like(pairwise_distances)
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            # Do not count the atom or its direct neighbors
            if np.abs(i - j) <= 1:
                continue
            # Set as alpha * (r_i + r_j)
            # see https://www.pnas.org/doi/10.1073/pnas.072665799
            clash_distance[i, j] = alpha * (
                VDW_RADII[atoms.element[i]] + VDW_RADII[atoms.element[j]]
            )
    is_clash = pairwise_distances <= clash_distance
    # Set the diagonal to be non-clashing - atoms do not clash with themselves
    is_clash[np.identity(len(atoms), dtype=bool)] = False
    if not np.all(is_clash == is_clash.T):
        raise ValueError(f"Clash matrix is not symmetric for {fname}")

    # Count the number of clashes
    n_clashes = np.sum(np.any(is_clash, axis=1))
    return n_clashes


def count_clashes_parallel(
    filenames: Collection[str], nthreads: int = mp.cpu_count()
) -> Dict[str, int]:
    """Parallelized calculation of clashes for a collection of pdb files."""
    with mp.Pool(nthreads) as pool:
        n_clashes = pool.map(count_clashes, tqdm(filenames), chunksize=10)
    retval = dict(zip(filenames, n_clashes))
    return retval


if __name__ == "__main__":
    import sys

    count_clashes_parallel(sys.argv[1:])
