"""
Script to sample from a trained diffusion model
"""
import multiprocessing
import os, sys
import argparse
import logging
import json
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch

# Import data loading code from main training script
from train import get_train_valid_test_sets
from annot_secondary_structures import make_ss_cooccurrence_plot

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import modelling
import beta_schedules
import sampling
import plotting
from datasets import NoisedAnglesDataset
from angles_and_coords import create_new_chain_nerf

# :)
SEED = int(
    float.fromhex("54616977616e20697320616e20696e646570656e64656e7420636f756e747279")
    % 10000
)

FT_NAME_MAP = {
    "phi": r"$\phi$",
    "psi": r"$\psi$",
    "omega": r"$\omega$",
    "tau": r"$\theta_1$",
    "CA:C:1N": r"$\theta_2$",
    "C:1N:1CA": r"$\theta_3$",
}


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
        train_only=False,
    )

    train_dset, valid_dset, test_dset = get_train_valid_test_sets(**dset_args)
    logging.info(
        f"Training dset contains features: {train_dset.feature_names} - angular {train_dset.feature_is_angular}"
    )
    return train_dset, valid_dset, test_dset


def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
    basename_prefix: str = "generated_",
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Writing sampled angles as PDB files to {outdir}")
    retval = []
    for i, samp in enumerate(final_sampled):
        fname = create_new_chain_nerf(
            os.path.join(outdir, f"{basename_prefix}{i}.pdb"), samp
        )
        assert fname
        retval.append(fname)
    return retval


def plot_ramachandran(
    phi_values, psi_values, fname: str, annot_ss: bool = False, title: str = ""
):
    """Create Ramachandran plot for phi_psi"""
    fig = plotting.plot_joint_kde(
        phi_values,
        psi_values,
    )
    ax = fig.axes[0]
    if annot_ss:
        # https://matplotlib.org/stable/tutorials/text/annotations.html
        ram_annot_arrows = dict(
            facecolor="black", shrink=0.05, headwidth=6.0, width=1.5
        )
        ax.annotate(
            r"$\alpha$ helix, LH",
            xy=(1.2, 0.5),
            xycoords="data",
            xytext=(2.0, 1.2),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.annotate(
            r"$\alpha$ helix, RH",
            xy=(-1.1, -0.6),
            xycoords="data",
            xytext=(-1.9, -1.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="right",
            verticalalignment="center",
        )
        ax.annotate(
            r"$\beta$ sheet",
            xy=(-1.67, 2.25),
            xycoords="data",
            xytext=(-0.9, 3.75),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
        )
    ax.set_xlabel("$\phi", fontsize=12)
    ax.set_ylabel("$\psi", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    fig.savefig(fname, bbox_inches="tight")


def plot_distribution_overlap(
    values_dicts: Dict[str, np.ndarray],
    title: str = "Sampled distribution",
    fname: str = "",
    ax=None,
    show_legend: bool = True,
    **kwargs,
):
    """
    Plot the distribution overlap between the training and sampled values
    Additional arguments are given to ax.hist; for example, can specify
    histtype='step', cumulative=True
    to get a CDF plot
    """
    # Plot the distribution overlap
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    bins = 50
    for k, v in values_dicts.items():
        _n, bins, _pbatches = ax.hist(
            v,
            bins=bins,
            density=True,
            label=k,
            alpha=0.6,
            **kwargs,
        )
    ax.set(title=title)
    if show_legend:
        ax.legend()
    if fname:
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
        "-l",
        "--lengths",
        type=str,
        choices=["sample", "sweep"],
        default="sweep",
        help="Strategy for generating lengths of sequences. Sampled will sample training set lengths, sweep will sweep from 70-max length",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=512,
        help="Batch size to use when sampling. 256 consumes ~2GB of GPU memory, 512 ~3.5GB",
    )
    parser.add_argument(
        "--fullhistory",
        action="store_true",
        help="Store full history, not just final structure",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser


def main() -> None:
    """Run the script"""
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

    # Load the dataset based on training args
    train_dset, _, test_dset = build_datasets(training_args)
    # Fetch values for training distribution
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]
    train_values = [
        select_by_attn(train_dset.dset.__getitem__(i, ignore_zero_center=True))
        for i in range(len(train_dset))
    ]
    train_values_stacked = torch.cat(train_values, dim=0).cpu().numpy()

    test_values = [
        select_by_attn(test_dset.dset.__getitem__(i, ignore_zero_center=True))
        for i in range(len(test_dset))
    ]
    test_values_stacked = torch.cat(test_values, dim=0).cpu().numpy()

    # Plot ramachandran plot for the training distribution
    # Default figure size is 6.4x4.8 inches
    phi_idx = test_dset.feature_names["angles"].index("phi")
    psi_idx = test_dset.feature_names["angles"].index("psi")
    plot_ramachandran(
        test_values_stacked[:5000, phi_idx],
        test_values_stacked[:5000, psi_idx],
        annot_ss=True,
        fname=plotdir / "ramachandran_test_annot.pdf",
    )

    # Load the model
    model_snapshot_dir = outdir / "model_snapshot"
    model = modelling.BertForDiffusion.from_dir(
        args.model, copy_to=model_snapshot_dir
    ).to(torch.device(args.device))

    # Perform sampling
    torch.manual_seed(args.seed)
    if args.lengths == "sample":
        sampled = sampling.sample(
            model, train_dset, n=args.num, batch_size=args.batchsize
        )
    elif args.lengths == "sweep":
        sampled = sampling.sample(
            model,
            train_dset,
            n=10,
            sweep_lengths=(50, test_dset.dset.pad),
            batch_size=args.batchsize,
        )
    else:
        raise ValueError(f"Unrecognized length mode: {args.lengths}")
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=train_dset.feature_names["angles"])
        for s in final_sampled
    ]

    # Write the raw sampled items to csv files
    sampled_angles_folder = outdir / "sampled_angles"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    logging.info(f"Writing sampled angles to {sampled_angles_folder}")
    for i, s in enumerate(sampled_dfs):
        s.to_csv(sampled_angles_folder / f"generated_{i}.csv.gz")
    # Write the sampled angles as pdb files
    pdb_files = write_preds_pdb_folder(sampled_dfs, outdir / "sampled_pdb")

    # If full history is specified, create a separate directory and write those files
    if args.fullhistory:
        # Write the angles
        full_history_angles_dir = sampled_angles_folder / "sample_history"
        os.makedirs(full_history_angles_dir)
        full_history_pdb_dir = outdir / "sampled_pdb/sample_history"
        os.makedirs(full_history_pdb_dir)
        # sampled is a list of np arrays
        for i, sampled_series in enumerate(sampled):
            snapshot_dfs = [
                pd.DataFrame(snapshot, columns=train_dset.feature_names["angles"])
                for snapshot in sampled_series
            ]
            # Write the angles
            ith_angle_dir = full_history_angles_dir / f"generated_{i}"
            os.makedirs(ith_angle_dir, exist_ok=True)
            for timestep, snapshot_df in enumerate(snapshot_dfs):
                snapshot_df.to_csv(
                    ith_angle_dir / f"generated_{i}_timestep_{timestep}.csv.gz"
                )
            # Write the pdb files
            ith_pdb_dir = full_history_pdb_dir / f"generated_{i}"
            write_preds_pdb_folder(
                snapshot_dfs, ith_pdb_dir, basename_prefix=f"generated_{i}_timestep_"
            )

    # Generate histograms of sampled angles -- separate plots, and a combined plot
    # For calculating angle distributions
    multi_fig, multi_axes = plt.subplots(
        dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True
    )
    step_multi_fig, step_multi_axes = plt.subplots(
        dpi=300, nrows=2, ncols=3, figsize=(14, 6), sharex=True
    )
    final_sampled_stacked = np.vstack(final_sampled)
    for i, ft_name in enumerate(test_dset.feature_names["angles"]):
        orig_values = test_values_stacked[:, i]
        samp_values = final_sampled_stacked[:, i]

        ft_name_readable = FT_NAME_MAP[ft_name]

        # Plot single plots
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle distribution - {ft_name_readable}",
            fname=plotdir / f"dist_{ft_name}.pdf",
        )
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle CDF - {ft_name_readable}",
            histtype="step",
            cumulative=True,
            fname=plotdir / f"cdf_{ft_name}.pdf",
        )

        # Plot combo plots
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle distribution - {ft_name_readable}",
            ax=multi_axes.flatten()[i],
            show_legend=i == 0,
        )
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"Sampled angle CDF - {ft_name_readable}",
            cumulative=True,
            histtype="step",
            ax=step_multi_axes.flatten()[i],
            show_legend=i == 0,
        )
    multi_fig.savefig(plotdir / "dist_combined.pdf", bbox_inches="tight")
    step_multi_fig.savefig(plotdir / "cdf_combined.pdf", bbox_inches="tight")

    # Generate ramachandran plot for sampled angles
    plot_ramachandran(
        final_sampled_stacked[:5000, phi_idx],
        final_sampled_stacked[:5000, psi_idx],
        fname=plotdir / "ramachandran_generated.pdf",
    )

    # Generate plots of secondary structure co-occurrence
    make_ss_cooccurrence_plot(
        pdb_files,
        str(outdir / "plots" / "ss_cooccurrence_sampled.pdf"),
        threads=multiprocessing.cpu_count(),
    )
    make_ss_cooccurrence_plot(
        test_dset.filenames,
        str(outdir / "plots" / "ss_cooccurrence_test.pdf"),
        max_seq_len=test_dset.dset.pad,
        threads=multiprocessing.cpu_count(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
