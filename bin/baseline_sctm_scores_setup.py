"""
Short script to set up files for calculating baseline for scTM scores. This
baseline is constructed by taking naturally occurring human proteins and
seeing how many of them have passing scTM scores if we run them through the
same process of ESM inverse folding > omegafold. 

Setup constitutes creating a new folder that contains symlinks to the files
we will use to construct baselines. After this folder is constructed, run the
following commands:
> python bin/baseline_sctm_scores_setup.py -n 512 -o sctm_baseline_real_pdbs
> conda activate inverse
> python bin/pdb_to_residues_esm.py sctm_baseline_real_pdbs -o sctm_baseline_esm_residues
> conda activate omegafold
> python bin/omegafold_across_gpus.py sctm_baseline_esm_residues/*.pdb -o sctm_baseline_omegafold_predictions
> conda activate protdiff
> python bin/omegafold_self_tm.py -p sctm_baseline_esm_residues -f sctm_baseline_omegafold_predictions -o baseline_sctm_scores.json -p baseline_sctm_scores.pdf
"""
import os
import argparse

import numpy as np

from train import get_train_valid_test_sets

#  :)
SEED = int(float.fromhex("54616977616e2069732061206672656520636f756e7472792e") % 10000)


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=512,
        help="Number of examples to use in building baseline",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.join(os.getcwd(), "sctm_baseline_real_pdbs"),
        help="Outdir to place symlinks to files",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    # Get the test set. We really just need the sequence lenghts and the filenames
    # Timesteps and variance schedule don't really matter here
    _, _, test_set = get_train_valid_test_sets(
        angles_definitions="canonical-full-angles",
        max_seq_len=128,
        seq_trim_strategy="discard",
        timesteps=1000,
        variance_schedule="cosine",
    )

    # Choose 512 random sequences
    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(len(test_set), size=args.num, replace=False)
    test_dset_fillenames = [test_set.filenames[i] for i in idx]

    # Create symlinks
    os.makedirs(args.outdir, exist_ok=False)
    for fname in test_dset_fillenames:
        assert os.path.isfile(fname)
        dest_fname = os.path.join(args.outdir, os.path.basename(fname))
        if not dest_fname.endswith(".pdb"):
            dest_fname += ".pdb"
        os.symlink(fname, dest_fname)


if __name__ == "__main__":
    main()
