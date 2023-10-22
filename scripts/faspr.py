"""
Wrapper to run FASPR side chain packing on a pdb file.

https://github.com/tommyhuangthu/FASPR
"""

import os
import argparse
from glob import glob
from collections import defaultdict
import itertools
import tempfile
import subprocess
import shlex
import logging
import multiprocessing as mp
from typing import *

from foldingdiff.tmalign import match_files


logging.basicConfig(level=logging.INFO)
FASPR_BIN = os.path.expanduser("~/software/FASPR/FASPR")
assert os.path.isfile(FASPR_BIN)


def read_fasta(fasta_fname: str) -> Dict[str, str]:
    """Read the fasta file as a dict mapping identifiers to sequences."""
    retval = defaultdict(str)
    with open(fasta_fname, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                identifier = line[1:]
            else:
                retval[identifier] += line
    return retval


def run_faspr(input_pdb: str, input_fasta: str, output_pdb: str) -> str:
    """Run FASPR and write structure to output_pdb."""
    # sequence should be written a single line of one-letter alphabet; we read
    # in the fasta file and rewrite the sequences to adhere to this format
    fasta_seqs = read_fasta(input_fasta)
    assert len(fasta_seqs) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the fasta file
        seq_fname = os.path.join(tmpdir, "seq.txt")
        with open(seq_fname, "w") as sink:
            sink.write(fasta_seqs.popitem()[1] + "\n")

        # Run FASPR
        cmd = f"{FASPR_BIN} -i {input_pdb} -s {seq_fname} -o {output_pdb}"
        subprocess.check_call(
            shlex.split(cmd), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
    return output_pdb


def build_parser() -> argparse.ArgumentParser:
    """Build a basic CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Wrapper for running FASPR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_pdb_dir", type=str, help="Input dir containing .pdb files"
    )
    parser.add_argument(
        "input_fasta_dir", type=str, help="Input dir containing .fasta files"
    )
    parser.add_argument("output_pdb_dir", type=str, help="Output dir for .pdb files")
    return parser


def main():
    """Run script."""
    args = build_parser().parse_args()

    query2fasta = match_files(
        glob(os.path.join(args.input_pdb_dir, "*.pdb")),
        glob(os.path.join(args.input_fasta_dir, "*.fasta")),
        strategy="prefix",
    )
    logging.info(
        f"Matched {len(query2fasta)} pdb files to {len(list(itertools.chain.from_iterable(query2fasta.values())))} fasta files"
    )

    # Make the output directory
    os.makedirs(args.output_pdb_dir, exist_ok=True)

    # Create arg tuples (pdb, fasta, out_pdb)
    arg_tuples = []
    for query_pdb, fasta_files in query2fasta.items():
        for fasta_file in fasta_files:
            arg_tuples.append(
                (
                    query_pdb,
                    fasta_file,
                    os.path.join(
                        args.output_pdb_dir,
                        os.path.basename(fasta_file).replace(".fasta", ".pdb"),
                    ),
                )
            )

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(run_faspr, arg_tuples, chunksize=10)


if __name__ == "__main__":
    main()
