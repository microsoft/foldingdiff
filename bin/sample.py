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
import mpl_scatter_density
from matplotlib import pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import torch
from huggingface_hub import snapshot_download

# Import data loading code from main training script
from train import get_train_valid_test_sets
from annot_secondary_structures import make_ss_cooccurrence_plot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from foldingdiff import modelling
from foldingdiff import sampling
from foldingdiff import plotting
from foldingdiff.datasets import AnglesEmptyDataset, NoisedAnglesDataset
from foldingdiff.angles_and_coords import create_new_chain_nerf
from foldingdiff import utils

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
    model_dir: Path, load_actual: bool = True
) -> Tuple[NoisedAnglesDataset, NoisedAnglesDataset, NoisedAnglesDataset]:
    """
    Build datasets given args again. If load_actual is given, the load the actual datasets
    containing actual values; otherwise, load a empty shell that provides the same API for
    faster generation.
    """
    with open(model_dir / "training_args.json") as source:
        training_args = json.load(source)
    # Build args based on training args
    if load_actual:
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
    else:
        mean_file = model_dir / "training_mean_offset.npy"
        placeholder_dset = AnglesEmptyDataset(
            feature_set_key=training_args["angles_definitions"],
            pad=training_args["max_seq_len"],
            mean_offset=None if not mean_file.exists() else np.load(mean_file),
        )
        noised_dsets = [
            NoisedAnglesDataset(
                dset=placeholder_dset,
                dset_key="coords"
                if training_args["angles_definitions"] == "cart-coords"
                else "angles",
                timesteps=training_args["timesteps"],
                exhaustive_t=False,
                beta_schedule=training_args["variance_schedule"],
                nonangular_variance=1.0,
                angular_variance=training_args["variance_scale"],
            )
            for _ in range(3)
        ]
        return noised_dsets


def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
    basename_prefix: str = "generated_",
    threads: int = multiprocessing.cpu_count(),
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(
        f"Writing sampled angles as PDB files to {outdir} using {threads} threads"
    )
    # Create the pairs of arguments
    arg_tuples = [
        (os.path.join(outdir, f"{basename_prefix}{i}.pdb"), samp)
        for i, samp in enumerate(final_sampled)
    ]
    # Write in parallel
    with multiprocessing.Pool(threads) as pool:
        files_written = pool.starmap(create_new_chain_nerf, arg_tuples)

    return files_written


def plot_ramachandran(
    phi_values,
    psi_values,
    fname: str,
    annot_ss: bool = False,
    title: str = "",
    plot_type: Literal["kde", "density_heatmap"] = "density_heatmap",
):
    """Create Ramachandran plot for phi_psi"""
    if plot_type == "kde":
        fig = plotting.plot_joint_kde(
            phi_values,
            psi_values,
        )
        ax = fig.axes[0]
        ax.set_xlim(-3.67, 3.67)
        ax.set_ylim(-3.67, 3.67)
    elif plot_type == "density_heatmap":
        fig = plt.figure(dpi=800)
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        norm = ImageNormalize(vmin=0.0, vmax=650, stretch=LogStretch())
        ax.scatter_density(phi_values, psi_values, norm=norm, cmap=plt.cm.Blues)
    else:
        raise NotImplementedError(f"Cannot plot type: {plot_type}")
    if annot_ss:
        # https://matplotlib.org/stable/tutorials/text/annotations.html
        ram_annot_arrows = dict(
            facecolor="black", shrink=0.05, headwidth=6.0, width=1.5
        )
        ax.annotate(
            r"$\alpha$ helix, LH",
            xy=(1.2, 0.5),
            xycoords="data",
            xytext=(1.7, 1.2),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\alpha$ helix, RH",
            xy=(-1.1, -0.6),
            xycoords="data",
            xytext=(-1.7, -1.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\beta$ sheet",
            xy=(-1.67, 2.25),
            xycoords="data",
            xytext=(-0.9, 2.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
    ax.set_xlabel("$\phi$ (radians)", fontsize=14)
    ax.set_ylabel("$\psi$ (radians)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    fig.savefig(fname, bbox_inches="tight")


def plot_distribution_overlap(
    values_dicts: Dict[str, np.ndarray],
    title: str = "Sampled distribution",
    fname: str = "",
    bins: int = 50,
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

    for k, v in values_dicts.items():
        if v is None:
            continue
        _n, bins, _pbatches = ax.hist(
            v,
            bins=bins,
            label=k,
            density=True,
            **kwargs,
        )
    if title:
        ax.set_title(title, fontsize=16)
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
        "-m",
        "--model",
        type=str,
        default="wukevin/foldingdiff_cath",
        help="Path to model directory, or a repo identifier on huggingface hub. Should contain training_args.json, config.json, and models folder at a minimum.",
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=os.getcwd(), help="Path to output directory"
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="Number of examples to generate *per length*",
    )
    parser.add_argument(
        "-l",
        "--lengths",
        type=int,
        nargs=2,
        default=[50, 128],
        help="Range of lengths to sample from",
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
    parser.add_argument(
        "--testcomparison", action="store_true", help="Run comparison against test set"
    )
    parser.add_argument("--nopsea", action="store_true", help="Skip PSEA calculations")
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
    # Be extra cautious so we don't overwrite any results
    assert not os.listdir(outdir), f"Expected {outdir} to be empty!"

    # Download the model if it was given on modelhub
    if utils.is_huggingface_hub_id(args.model):
        logging.info(f"Detected huggingface repo ID {args.model}")
        dl_path = snapshot_download(args.model)  # Caching is automatic
        assert os.path.isdir(dl_path)
        logging.info(f"Using downloaded model at {dl_path}")
        args.model = dl_path

    plotdir = outdir / "plots"
    os.makedirs(plotdir, exist_ok=True)

    # Load the dataset based on training args
    train_dset, _, test_dset = build_datasets(
        Path(args.model), load_actual=args.testcomparison
    )
    phi_idx = test_dset.feature_names["angles"].index("phi")
    psi_idx = test_dset.feature_names["angles"].index("psi")
    # Fetch values for training distribution
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]

    if args.testcomparison:
        test_values = [
            select_by_attn(test_dset.dset.__getitem__(i, ignore_zero_center=True))
            for i in range(len(test_dset))
        ]
        test_values_stacked = torch.cat(test_values, dim=0).cpu().numpy()

        # Plot ramachandran plot for the training distribution
        # Default figure size is 6.4x4.8 inches
        plot_ramachandran(
            test_values_stacked[:, phi_idx],
            test_values_stacked[:, psi_idx],
            annot_ss=True,
            fname=plotdir / "ramachandran_test_annot.pdf",
        )
    else:
        test_values_stacked = None

    # Load the model
    model_snapshot_dir = outdir / "model_snapshot"
    model = modelling.BertForDiffusionBase.from_dir(
        args.model, copy_to=model_snapshot_dir
    ).to(torch.device(args.device))

    # Checks
    sweep_min_len, sweep_max_len = args.lengths
    assert sweep_min_len < sweep_max_len
    assert sweep_max_len <= train_dset.dset.pad

    # Perform sampling
    torch.manual_seed(args.seed)
    sampled = sampling.sample(
        model,
        train_dset,
        n=args.num,
        sweep_lengths=(sweep_min_len, sweep_max_len),
        batch_size=args.batchsize,
    )
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
        orig_values = (
            test_values_stacked[:, i] if test_values_stacked is not None else None
        )
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
        final_sampled_stacked[:, phi_idx],
        final_sampled_stacked[:, psi_idx],
        fname=plotdir / "ramachandran_generated.pdf",
    )

    # Generate plots of secondary structure co-occurrence
    if not args.nopsea:
        make_ss_cooccurrence_plot(
            pdb_files,
            str(outdir / "plots" / "ss_cooccurrence_sampled.pdf"),
            threads=multiprocessing.cpu_count(),
        )
        if args.testcomparison:
            make_ss_cooccurrence_plot(
                test_dset.filenames,
                str(outdir / "plots" / "ss_cooccurrence_test.pdf"),
                max_seq_len=test_dset.dset.pad,
                threads=multiprocessing.cpu_count(),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
