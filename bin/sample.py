"""
Script to sample from a trained diffusion model
"""
import os, sys
import argparse
import logging
import json
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch

# Import data loading code from main training script
from train import get_train_valid_test_sets

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import modelling
import beta_schedules
import sampling
import plotting
from datasets import NoisedAnglesDataset, CathCanonicalAnglesDataset
from angles_and_coords import create_new_chain_nerf
import tmalign

# :)
SEED = int(
    float.fromhex("54616977616e20697320616e20696e646570656e64656e7420636f756e747279")
    % 10000
)


def build_datasets(
    training_args: Dict[str, Any]
) -> Tuple[NoisedAnglesDataset, NoisedAnglesDataset, NoisedAnglesDataset]:
    """
    Build datasets given args again
    """
    # Build args based on training args
    dset_args = dict(
        timesteps=training_args["timesteps"],
        variance_schedule=training_args["variance_schedule"],
        max_seq_len=training_args["max_seq_len"],
        min_seq_len=training_args["min_seq_len"],
        var_scale=training_args["variance_scale"],
        syn_noiser=training_args["syn_noiser"],
        exhaustive_t=training_args["exhaustive_validation_t"],
        single_angle_debug=training_args["single_angle_debug"],
        single_time_debug=training_args["single_timestep_debug"],
        toy=training_args["subset"],
        angles_definitions=training_args["angles_definitions"],
        zero_center=training_args["zero_center"],
        train_only=True,
    )

    train_dset, valid_dset, test_dset = get_train_valid_test_sets(**dset_args)
    logging.info(
        f"Training dset contains features: {train_dset.feature_names} - angular {train_dset.feature_is_angular}"
    )
    return train_dset, valid_dset, test_dset


def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Writing sampled anlges as PDB files to {outdir}")
    retval = []
    for i, samp in enumerate(final_sampled):
        fname = create_new_chain_nerf(os.path.join(outdir, f"generated_{i}.pdb"), samp)
        assert fname
        retval.append(fname)
    return retval


def plot_distribution_overlap(
    train_values: np.ndarray, sampled_values: np.ndarray, ft_name: str, fname: str
):
    """
    Plot the distribution overlap between the training and sampled values
    """
    # Plot the distribution overlap
    logging.info(f"Plotting distribution overlap for {ft_name}")
    fig, ax = plt.subplots(dpi=300)
    sns.histplot(
        train_values,
        bins=40,
        stat="proportion",
        ax=ax,
        label="Training",
        color="tab:blue",
        alpha=0.6,
    )
    sns.histplot(
        sampled_values,
        bins=40,
        stat="proportion",
        ax=ax,
        label="Sampled",
        color="tab:orange",
        alpha=0.6,
    )
    ax.set(title=f"Sampled distribution - {ft_name}")
    ax.legend()
    fig.savefig(fname, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to model directory. Should contain training_args.json, config.json, and models folder at a minimum.",
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=os.getcwd(), help="Path to output directory"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=512, help="Number of examples to generate"
    )
    parser.add_argument(
        "--legacy", action="store_true", help="Use legacy model loading code"
    )
    parser.add_argument(
        "--skiptm",
        action="store_true",
        help="Skip calculation of TM scores against training set",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.info(f"Creating {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    outdir = Path(args.outdir)
    assert not os.listdir(outdir), f"Expected {outdir} to be empty!"

    plotdir = outdir / "plots"
    os.makedirs(plotdir, exist_ok=True)

    with open(os.path.join(args.model, "training_args.json")) as source:
        training_args = json.load(source)

    # Reproduce the beta schedule
    beta_values = beta_schedules.get_variance_schedule(
        training_args["variance_schedule"],
        training_args["timesteps"],
    )
    alpha_beta_values = beta_schedules.compute_alphas(beta_values)
    alpha_beta_values.keys()

    # Load the dataset based on training args
    train_dset, valid_dset, test_dset = build_datasets(training_args)
    # Fetch values for training distribution
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]
    train_values = [
        select_by_attn(train_dset.dset.__getitem__(i, ignore_zero_center=True))
        for i in range(len(train_dset))
    ]
    train_values_stacked = torch.cat(train_values, dim=0).cpu().numpy()

    # Plot ramachandran plot for the training distribution
    phi_idx = train_dset.feature_names["angles"].index("phi")
    psi_idx = train_dset.feature_names["angles"].index("psi")
    plotting.plot_joint_kde(
        train_values_stacked[:5000, phi_idx],
        train_values_stacked[:5000, psi_idx],
        xlabel="$\phi$",
        ylabel="$\psi$",
        title="Ramachandran plot, training",
        fname=plotdir / "ramachandran_train.pdf",
    )

    # Load the model
    model = modelling.BertForDiffusion.from_dir(args.model, legacy=args.legacy).to(
        torch.device(args.device)
    )

    # Perform sampling
    torch.manual_seed(args.seed)
    sampled = sampling.sample(model, train_dset, n=args.num)
    final_sampled = [s[-1] for s in sampled]
    final_sampled_stacked = np.vstack(final_sampled)

    # Write the raw sampled items to np files
    sampled_angles_folder = outdir / "sampled_angles"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    logging.info(f"Writing sampled angles to {sampled_angles_folder}")
    for i, s in enumerate(sampled):
        np.save(sampled_angles_folder / f"generated_{i}.npy", s)

    # Generate histograms of sampled angles
    for i, ft_name in enumerate(train_dset.feature_names["angles"]):
        orig_values = train_values_stacked[:, i]
        samp_values = final_sampled_stacked[:, i]
        plot_distribution_overlap(
            orig_values, samp_values, ft_name, plotdir / f"dist_{ft_name}.pdf"
        )

    # Generate ramachandran plot for sampled angles
    plotting.plot_joint_kde(
        final_sampled_stacked[:5000, phi_idx],
        final_sampled_stacked[:5000, psi_idx],
        xlabel="$\phi$",
        ylabel="$\psi$",
        title="Ramachandran plot, generated",
        fname=plotdir / "ramachandran_generated.pdf",
    )

    # Write the sampled angles as pdb files
    sampled_dfs = [
        pd.DataFrame(s, columns=train_dset.feature_names["angles"])
        for s in final_sampled
    ]
    pdb_files = write_preds_pdb_folder(sampled_dfs, outdir / "sampled_pdb")

    if not args.skiptm:
        logging.info(f"Done writing main outputs! Calculating tm scores...")
        all_tm_scores = {}
        for i, fname in enumerate(pdb_files):
            samp_name = os.path.splitext(os.path.basename(fname))[0]
            tm_score = tmalign.max_tm_across_refs(fname, train_dset.dset.filenames)
            all_tm_scores[samp_name] = tm_score
        with open(outdir / "tm_scores.json", "w") as sink:
            json.dump(all_tm_scores, sink, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
