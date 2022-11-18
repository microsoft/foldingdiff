"""
Script to make the inputs for alphafold to run on sherlock

Copy fasta files and rename them to a3m files (this skips MSA)
"""
import os
import shutil
import logging
import argparse
from pathlib import Path

import numpy as np


def build_parser():
    """
    Build a CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fastadir", type=str, help="Directory containing fasta files")
    parser.add_argument(
        "outdir", type=str, help="Directory to write groups of a3m files to"
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=8,
        help="Number of subdirectories (jobs) to split across",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    fastadir = Path(args.fastadir)
    outdir = Path(args.outdir)
    num = args.num

    os.makedirs(outdir, exist_ok=True)

    fastas = list(fastadir.glob("*.fasta"))
    assert fastas
    logging.info(f"Found {len(fastas)} fasta files in {fastadir}")

    # Randomly split the files
    idx = np.arange(len(fastas))
    rng = np.random.default_rng(seed=1234)
    rng.shuffle(idx)
    idx_split = np.array_split(idx, num)

    for i, idx_chunk in enumerate(idx_split):
        subdir = outdir / f"af_job_{i}" / "inputs"
        os.makedirs(subdir, exist_ok=True)
        for j in idx_chunk:
            fname = fastas[j]
            newname = subdir / fname.name.replace(".fasta", ".a3m")
            logging.debug(f"Copying {fname} -> {newname}")
            shutil.copy(fname, newname)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
