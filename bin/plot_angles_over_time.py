"""

"""

import os, sys
import logging
import argparse
import glob
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def timestep_from_fname(fname: str) -> int:
    """
    Get the timestep from the given filename

    >>> timestep_from_fname("generated_0_timestep_0.csv.gz")
    0
    >>> timestep_from_fname("generated_0_timestep_99.csv.gz")
    99
    >>> timestep_from_fname("/foo/bar/generated_0_timestep_99.csv.gz")
    99
    """
    return int(os.path.basename(fname).split(".")[0].split("_")[-1])


def get_angle_files(dirname: str, subset_to: Optional[str] = None) -> List[List[str]]:
    """
    Return a list of files that contain sampled angles, one list for
    each timestep. This folder should be

    If subset_to is provided (e.g., "generated_0"), then only return
    files corresponding to the given generated example.
    """
    assert os.path.isdir(dirname)
    dirs_to_search = [subset_to] if subset_to else os.listdir(dirname)
    per_generation_filenames = []
    for folder in dirs_to_search:
        if not os.path.isdir(os.path.join(dirname, folder)) or not folder.startswith(
            "generated_"
        ):
            continue

        pattern = os.path.join(dirname, folder, "generated_*_timestep_*.csv.gz")
        files = sorted(
            glob.glob(pattern),
            key=timestep_from_fname,
        )
        assert files, f"Found no files with {pattern}"
        per_generation_filenames.append(files)

    retval = list(zip(*per_generation_filenames))
    return retval


def build_parser():
    """
    Build a basic CLI parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", help="Directory containing sampled angles history")
    parser.add_argument("outdir", help="Directory to write plots to")
    parser.add_argument(
        "--subset",
        default="",
        type=str,
        required=False,
        help="Subset to the given generation prefix (e.g., generated_0)",
    )
    parser.add_argument("--nolabel", action="store_true", help="Do not label plots")
    parser.add_argument("-a", "--angle", default="psi", help="Angle to plot")
    parser.add_argument(
        "-t",
        "--timesteps",
        nargs="+",
        type=int,
        help="Timesteps to plot",
        default=[0, 199, 399, 599, 799, 999],
    )
    return parser


def main(
    dirname: str,
    outdir: str,
    angle_to_plot: str = "psi",
    timesteps_to_plot: List[int] = [0, 199, 399, 599, 799, 999],
):
    """
    Run the script
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    angle_files = get_angle_files(dirname, subset_to=args.subset)
    assert angle_files, f"Found no files in {os.path.abspath(dirname)}"
    for timestep_files in angle_files:
        timestep = set([timestep_from_fname(f) for f in timestep_files])
        assert len(timestep) == 1
        timestep = timestep.pop()
        if timestep not in timesteps_to_plot:
            continue

        # Read in all values
        values = []
        for fname in timestep_files:
            df = pd.read_csv(fname, index_col=0)
            values.extend(df[angle_to_plot].values.tolist())
        values = np.array(values)

        # Plot a histogram of all values in that timestep
        fig, ax = plt.subplots()
        ax.hist(
            values,
            bins=np.linspace(-np.pi, np.pi, num=60),
            edgecolor="black",
            alpha=0.67,
            density=True,
        )
        ax.set(
            xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            xticklabels=[
                r"$-\pi$",
                r"$-\frac{\pi}{2}$",
                "$0$",
                r"$\frac{\pi}{2}$",
                r"$\pi$",
            ],
        )
        if not args.nolabel:
            ax.set(
                title=f"$\{angle_to_plot}$ ($t = {timestep}$)",
                xlabel="Angle (radians)",
                ylabel="Normalized frequency",
            )
        # For saving metadata
        # https://gist.github.com/SamWolski/6a53bf12a84cde17bc37b103ca095b30
        fig.savefig(
            os.path.join(outdir, f"{angle_to_plot}_timestep_{timestep}.pdf"),
            bbox_inches="tight",
            metadata={
                "Author": "Kevin E. Wu",
                "Producer": " ".join(sys.argv[:]),
                "Subject": "FoldingDiff",
            },
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import doctest

    doctest.testmod()
    args = build_parser().parse_args()
    main(
        dirname=os.path.abspath(args.dirname),
        outdir=args.outdir,
        angle_to_plot=args.angle,
        timesteps_to_plot=args.timesteps,
    )
