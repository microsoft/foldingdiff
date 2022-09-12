"""
Compute the maximum TM score against training set
"""
import logging
import os, sys
from glob import glob
from pathlib import Path
import argparse

from sample import compute_training_tm_scores
from datasets import CathCanonicalAnglesDataset

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dirname",
        type=str,
        default=os.path.join(os.getcwd(), "sampled_pdb"),
        help="Directory of generated PDB structures",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    assert os.path.isdir(args.dirname)
    generated_pdbs = glob(os.path.join(args.dirname, "*.pdb"))
    assert generated_pdbs
    logging.info(f"Found {len(generated_pdbs)} generated structures")

    # we only need the filenames from the training dataset so it doesn't really matter
    # what specific parameters we use to initialize it. The only important parameters are
    # min_length, which is default to 40 and likely unchanged
    train_dset = CathCanonicalAnglesDataset(split="train")

    # Calculate scores
    compute_training_tm_scores(generated_pdbs, train_dset, Path(args.dirname))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
