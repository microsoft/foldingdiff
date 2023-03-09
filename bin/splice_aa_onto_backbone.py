"""
Script to splice the given AA nucleotide sequence onto the given backbone.
"""

import os
import argparse
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
    parser.add_argument("backbone", help="Backbone PDB file")
    parser.add_argument("aa_fasta", help="Fasta file containing AA sequence")
    parser.add_argument("output", help="Output PDB file")
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()
    assert os.path.isfile(args.backbone), f"Backbone file {args.backbone} not found"
    assert os.path.isfile(
        args.aa_fasta
    ), f"Amino acid fasta file {args.aa_fasta} not found"

    aa_dict = read_fasta(args.aa_fasta)
    assert len(aa_dict) == 1, "Only one sequence allowed in fasta file"
    _aa_name, aa_seq = aa_dict.popitem()

    ac.add_sidechains_to_backbone(args.backbone, aa_seq, args.output)


if __name__ == "__main__":
    main()
