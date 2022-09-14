"""
Count secondary structures in a PDB file as determined by p-sea

https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
"""

import os
import logging
import warnings
import functools
import multiprocessing as mp
import argparse
from itertools import groupby
from collections import Counter
from typing import Tuple, Collection, Literal

import numpy as np
from matplotlib import pyplot as plt

import biotite.structure as struc
from biotite.application import dssp
from biotite.structure.io.pdb import PDBFile

SSE_BACKEND = Literal["dssp", "psea"]


def count_structures_in_pdb(
    fname: str, backend: SSE_BACKEND = "psea"
) -> Tuple[int, int]:
    """Count the secondary structures (# alpha, # beta) in the given pdb file"""
    assert os.path.exists(fname)

    # Get the secondary structure
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    source = PDBFile.read(fname)
    if source.get_model_count() > 1:
        return (-1, -1)
    source_struct = source.get_structure()[0]
    chain_ids = np.unique(source_struct.chain_id)
    assert len(chain_ids) == 1
    chain_id = chain_ids[0]

    if backend == "psea":
        # a = alpha helix, b = beta sheet, c = coil
        ss = struc.annotate_sse(source_struct, chain_id)
        # https://stackoverflow.com/questions/6352425/whats-the-most-pythonic-way-to-identify-consecutive-duplicates-in-a-list
        ss_grouped = [(k, sum(1 for _ in g)) for k, g in groupby(ss)]
        ss_counts = Counter([chain for chain, _ in ss_grouped])

        num_alpha = ss_counts["a"] if "a" in ss_counts else 0
        num_beta = ss_counts["b"] if "b" in ss_counts else 0
    elif backend == "dssp":
        # https://www.biotite-python.org/apidoc/biotite.application.dssp.DsspApp.html#biotite.application.dssp.DsspApp
        app = dssp.DsspApp(source_struct)
        app.start()
        app.join()
        ss = app.get_sse()
        ss_grouped = [(k, sum(1 for _ in g)) for k, g in groupby(ss)]
        ss_counts = Counter([chain for chain, _ in ss_grouped])

        num_alpha = ss_counts["H"] if "H" in ss_counts else 0
        num_beta = ss_counts["B"] if "B" in ss_counts else 0
    else:
        raise ValueError(f"Unrecognized backend: {backend}")
    logging.debug(f"From {fname}:\t{num_alpha} {num_beta}")
    return num_alpha, num_beta


def make_ss_cooccurrence_plot(
    pdb_files: Collection[str],
    outpdf: str,
    backend: SSE_BACKEND = "psea",
    threads: int = 4,
):
    """ """
    logging.info(f"Calculating {len(pdb_files)} structures using {backend}")
    pfunc = functools.partial(count_structures_in_pdb, backend=backend)
    pool = mp.Pool(threads)
    alpha_beta_counts = list(pool.map(pfunc, pdb_files, chunksize=10))
    pool.close()
    pool.join()

    alpha_beta_counts = [p for p in alpha_beta_counts if p != (-1, -1)]
    alpha_counts, beta_counts = zip(*alpha_beta_counts)

    fig, ax = plt.subplots(dpi=300)
    h = ax.hist2d(alpha_counts, beta_counts, bins=np.arange(10), density=True)
    ax.set(
        xlabel="Number of alpha helices",
        ylabel="Number of beta sheets",
        title="Co-occurrence frequencies of secondary structure elements",
    )
    fig.colorbar(h[-1], ax=ax)
    fig.savefig(outpdf, bbox_inches="tight")


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pdbfile",
        type=str,
        nargs="+",
        help="PDB files to compute secondary structures for",
    )
    parser.add_argument(
        "outpdf",
        type=str,
        help="PDF file to write plot of secondary structure co-occurrence frequencies",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dssp", "psea"],
        default="dssp",
        help="Backend for calculating secondary structure",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4, help="Number of threads to use"
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    make_ss_cooccurrence_plot(args.pdbfile, args.outpdf, args.backend, args.threads)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
