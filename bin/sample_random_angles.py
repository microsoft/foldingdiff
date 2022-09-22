"""
Randomly sample angles in the test set to construct a baselines for scTM scores

We sample angles by
"""
import os
from pathlib import Path
import json
import logging
import argparse

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch

from sample import build_datasets, create_new_chain_nerf


def random_angles_to_pdb(
    length: int,
    rep: int,
    seed: int,
    dirname: Path,
    angle_set: pd.DataFrame,
):
    """
    Given a set of angles, randomly sample them and write outputs to various subdirectories
    within dirname
    """
    angles_folder = dirname / "sampled_angles"
    pdb_folder = dirname / "sampled_pdb"
    bname = f"random_angles_length_{length}_rep_{rep}"
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(angle_set), size=length, replace=True)  # With replacement

    # Create the angles
    angles = pd.DataFrame(angle_set.values[idx], columns=angle_set.columns)
    angles.to_csv(angles_folder / f"{bname}.csv.gz")

    # Create the PDB file
    pdb_fname = create_new_chain_nerf(str(pdb_folder / f"{bname}.pdb"), angles)
    assert pdb_fname


def build_parser():
    """Build an argument parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "training_json", type=str, help="Model training json (for recreating data sets)"
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=os.getcwd(),
        help="Directory to create output folders",
    )
    return parser


def main():
    """
    Run script
    """
    parser = build_parser()
    args = parser.parse_args()

    with open(args.training_json) as source:
        training_args = json.load(source)

    _, _, test_dset = build_datasets(training_args)

    # Get the test set values
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]
    test_values = [
        select_by_attn(test_dset.dset.__getitem__(i, ignore_zero_center=True))
        for i in range(len(test_dset))
    ]
    test_values_stacked = torch.cat(test_values, dim=0).cpu().numpy()
    logging.info(
        f"Test set array of values: {type(test_values_stacked), test_values_stacked.shape}"
    )
    test_values_stacked_df = pd.DataFrame(test_values_stacked, columns=test_dset.feature_names['angles'])

    dirname = Path(args.dir)
    angles_folder = dirname / "sampled_angles"
    pdb_folder = dirname / "sampled_pdb"
    os.makedirs(angles_folder, exist_ok=True)
    os.makedirs(pdb_folder, exist_ok=True)

    for l in tqdm(range(50, 128)):
        for i in range(10):
            random_angles_to_pdb(
                length=l,
                rep=i,
                seed=i + 6489,
                dirname=dirname,
                angle_set=test_values_stacked_df,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
