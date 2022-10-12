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
> python bin/omegafold_across_gpus.py sctm_baseline_esm_residues/*.fasta -o sctm_baseline_omegafold_predictions
> conda activate protdiff
> python bin/omegafold_self_tm.py -p sctm_baseline_real_pdbs -f sctm_baseline_omegafold_predictions -o baseline_sctm_scores
"""
import logging
import os
import argparse

import numpy as np
from tqdm.auto import tqdm

from train import get_train_valid_test_sets

from foldingdiff import angles_and_coords as ac

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
    parser.add_argument("--maxlen", type=int, default=128, help='Maximum length of structure to consider')
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

    # Filter out structures that are too long
    orig_num = len(test_set.filenames)
    test_set_fnames_filt = [f for f in test_set.filenames if ac.get_pdb_length(f) <= args.maxlen]
    logging.info(f"Retaining {len(test_set_fnames_filt)}/{orig_num} test set structures of length <= {args.maxlen}")

    # Choose random structures
    rng = np.random.default_rng(seed=SEED)
    idx = rng.choice(len(test_set_fnames_filt), size=args.num, replace=False)
    test_dset_fillenames = [test_set_fnames_filt[i] for i in idx]

    # For each, convert to angles and write the rebuilt coordinates
    os.makedirs(args.outdir, exist_ok=False)
    for fname in tqdm(test_dset_fillenames):
        assert os.path.isfile(fname)
        assert ac.get_pdb_length(fname) <= args.maxlen
        # Pull out the coords
        coords = ac.canonical_distances_and_dihedrals(fname, distances=[], angles=ac.EXHAUSTIVE_ANGLES)
        assert coords.shape[1] == 6

        dest_fname = os.path.join(args.outdir, os.path.basename(fname))
        if not dest_fname.endswith(".pdb"):
            dest_fname += ".pdb"
        
        # Write the reconstructed structure
        f = ac.create_new_chain_nerf(dest_fname, coords)
        assert f


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
