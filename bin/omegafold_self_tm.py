"""
Script for calculating self TM score given the input as 
"""

import argparse
import functools
from glob import glob
import os, sys
import logging
from pathlib import Path
import multiprocessing as mp
import json
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from biotite import structure as struc
from biotite.structure.io.pdb import PDBFile

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import tmalign


def get_sctm_score(orig_pdb: Path, folded_dirname: Path) -> float:
    """get the self-consistency tm score"""
    bname = os.path.splitext(os.path.basename(orig_pdb))[0] + "_esm_residues_*.pdb"
    folded_pdbs = glob(os.path.join(folded_dirname, bname))
    if not folded_pdbs:
        return np.nan
    return tmalign.max_tm_across_refs(orig_pdb, folded_pdbs, parallel=False)


def get_pdb_length(fname: str) -> int:
    """
    Get the length of the chain described in the PDB file
    """
    structure = PDBFile.read(fname)
    if structure.get_model_count() > 1:
        return -1
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    l = int(len(backbone) / 3)
    return l


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--predicted",
        type=str,
        default=os.path.join(os.getcwd(), "sampled_pdb"),
        help="Directory of generated backbone sequences",
    )
    parser.add_argument(
        "-f",
        "--folded",
        type=str,
        default=os.path.join(os.getcwd(), "omegafold_predictions"),
        help="Directory with predicted structures based on inverse folding residues",
    )
    parser.add_argument(
        "-o",
        "--outprefix",
        type=str,
        default=os.path.join(os.getcwd(), "sctm_scores"),
        help="Output prefix for files to write",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    assert os.path.isdir(args.folded)
    assert os.path.isdir(args.predicted)

    orig_predicted_backbones = glob(os.path.join(args.predicted, "*.pdb"))
    logging.info(
        f"Computing selfTM scores across {len(orig_predicted_backbones)} generated structures"
    )
    orig_predicted_backbone_lens = {
        os.path.splitext(os.path.basename(f))[0]: get_pdb_length(f)
        for f in orig_predicted_backbones
    }
    orig_predicted_backbone_names = [
        os.path.splitext(os.path.basename(f))[0] for f in orig_predicted_backbones
    ]

    # Match up the files
    pfunc = functools.partial(get_sctm_score, folded_dirname=Path(args.folded))
    pool = mp.Pool(mp.cpu_count())
    sctm_scores_raw = list(pool.map(pfunc, orig_predicted_backbones, chunksize=5))
    pool.close()
    pool.join()

    sctm_non_nan_idx = [i for i, val in enumerate(sctm_scores_raw) if ~np.isnan(val)]
    sctm_scores_mapping = {
        orig_predicted_backbone_names[i]: sctm_scores_raw[i] for i in sctm_non_nan_idx
    }
    sctm_scores = np.array(list(sctm_scores_mapping.values()))

    passing_num = np.sum(sctm_scores >= 0.5)
    logging.info(
        f"{len(sctm_scores)} entries with scores, {passing_num} passing 0.5 cutoff"
    )

    # Write the output
    logging.info(
        f"scTM score mean/median: {np.mean(sctm_scores), np.median(sctm_scores)}"
    )
    with open(args.outprefix + ".json", "w") as sink:
        json.dump(sctm_scores_mapping, sink, indent=4)

    # Create histogram of values
    fig, ax = plt.subplots()
    ax.hist(sctm_scores, bins=25, alpha=0.6)
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set(
        xlabel=f"scTM, $n={passing_num}$ are designable $(\geq 0.5$)",
        title=f"Self-consistency TM (scTM) scores, {len(sctm_scores)} generated protein backbones",
    )
    fig.savefig(args.outprefix + "_hist.pdf", bbox_inches="tight")

    # Create histogram of values by length
    sctm_scores_with_len = pd.DataFrame(
        [
            (sctm_scores_mapping[k], orig_predicted_backbone_lens[k])
            for k in sctm_scores_mapping.keys()
        ],
        columns=["scTM", "length_int"],
    )
    sctm_scores_with_len["length"] = [
        r"short ($\leq 70$ aa)" if l <= 70 else r"long ($> 70$ aa)"
        for l in sctm_scores_with_len["length_int"]
    ]

    fig, ax = plt.subplots(dpi=300)
    sns.histplot(sctm_scores_with_len, x="scTM", hue="length")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax.set(
        title=f"Self-consistency TM (scTM) scores, {len(sctm_scores)} generated protein backbones",
    )
    fig.savefig(args.outprefix + "_hist_by_len.pdf", bbox_inches="tight")

    # Create a jointplot of values if we can also find the training TM scores
    training_tm_scores_fname = os.path.join(args.predicted, "tm_scores.json")
    if os.path.isfile(training_tm_scores_fname):
        with open(training_tm_scores_fname) as source:
            training_tm_scores = json.load(source)
        shared_keys = [k for k in sctm_scores_mapping.keys() if k in training_tm_scores]
        logging.info(
            f"Found {len(shared_keys)} overlapped keys with training tm scores at {training_tm_scores_fname}"
        )
        # Pair them up and plot
        scores_df = pd.DataFrame(
            [(sctm_scores_mapping[k], training_tm_scores[k]) for k in shared_keys],
            columns=["scTM", "max training TM"],
        )

        jointgrid = sns.jointplot(scores_df, x="max training TM", y="scTM")
        for ax in (jointgrid.ax_joint, jointgrid.ax_marg_x):
            ax.axvline(0.5, color="grey", alpha=0.5, linestyle="--")
        for ax in (jointgrid.ax_joint, jointgrid.ax_marg_y):
            ax.axhline(0.5, color="grey", alpha=0.5, linestyle="--")
        jointgrid.savefig(args.outprefix + "_training_tm_scatter.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
