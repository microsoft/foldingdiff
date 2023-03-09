"""
Script to splice the given AA nucleotide sequence onto the given backbone.
"""

import os
import argparse
import multiprocessing as mp
from typing import Dict

from foldingdiff import angles_and_coords as ac


def read_fasta(filename: str) -> Dict[str, str]:
    """
    Read a fasta file into a dictionary
    """
    retval = {}
    header, curr_seq = None, ""
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Append the last item
                if header:
                    assert curr_seq
                    retval[header] = curr_seq
                header = line.strip(">").split()[0]
                curr_seq = ""
            else:
                assert header is not None
                curr_seq += line
    assert header and curr_seq
    retval[header] = curr_seq
    return retval


def build_parser():
    """
    Build a basic CLI parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", nargs="+", help="Backbone PDB file")
    parser.add_argument(
        "--fasta", type=str, required=True, help="Fasta file containing AA sequence"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory to write PDB files",
    )
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()
    for fname in args.backbone:
        assert os.path.isfile(fname), f"Backbone file {fname} not found"
    assert os.path.isfile(args.fasta), f"Amino acid fasta file {args.fasta} not found"

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    args.outdir = os.path.abspath(args.outdir)

    aa_dict = read_fasta(args.fasta)
    assert len(aa_dict) == 1, "Only one sequence allowed in fasta file"
    _aa_name, aa_seq = aa_dict.popitem()

    # For each, add the sidechains
    arg_tuples = [
        (
            backbone_fname,
            aa_seq,
            os.path.join(args.outdir, os.path.basename(backbone_fname)),
        )
        for backbone_fname in args.backbone
    ]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(ac.add_sidechains_to_backbone, arg_tuples, chunksize=10)


if __name__ == "__main__":
    main()
