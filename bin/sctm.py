"""
Script for calculating self consistency TM scores
"""

import argparse
import functools
from glob import glob
import os
import logging
from pathlib import Path
import multiprocessing as mp
import json
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

from biotite import structure as struc
from biotite.structure.io.pdb import PDBFile

from annot_secondary_structures import count_structures_in_pdb

from foldingdiff import tmalign
from foldingdiff.angles_and_coords import get_pdb_length


def get_sctm_score(orig_pdb: Path, folded_dirname: Path) -> Tuple[float, str]:
    """get the self-consistency tm score"""
    bname = os.path.splitext(os.path.basename(orig_pdb))[0] + "_*_residues_*.pdb"
    folded_pdbs = glob(os.path.join(folded_dirname, bname))
    assert len(folded_pdbs) <= 10  # We have never run more than 10 per before
    if len(folded_pdbs) > 8:
        folded_pdbs = folded_pdbs[:8]
    assert len(folded_pdbs) <= 8
    if len(folded_pdbs) < 8:
        logging.warning(
            f"Fewer than 8 (n={len(folded_pdbs)}) structures corresponding to {orig_pdb}"
        )
    if not folded_pdbs:
        return np.nan, ""
    return tmalign.max_tm_across_refs(orig_pdb, folded_pdbs, parallel=False)


def build_parser():
    """Build a CLI parser"""
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
    assert os.path.isdir(args.predicted), f"Directory not found: {args.predicted}"

    orig_predicted_backbones = glob(os.path.join(args.predicted, "*.pdb"))
    logging.info(
        f"Computing scTM scores across {len(orig_predicted_backbones)} generated structures"
    )
    orig_predicted_backbone_lens = {
        os.path.splitext(os.path.basename(f))[0]: get_pdb_length(f)
        for f in orig_predicted_backbones
    }
    orig_predicted_backbone_names = [
        os.path.splitext(os.path.basename(f))[0] for f in orig_predicted_backbones
    ]
    with mp.Pool(mp.cpu_count()) as pool:
        ss_counts = list(
            pool.map(count_structures_in_pdb, orig_predicted_backbones, chunksize=10)
        )
        orig_predicted_secondary_structs = {
            os.path.splitext(os.path.basename(f))[0]: s
            for f, s in zip(orig_predicted_backbones, ss_counts)
        }

    # Match up the files
    pfunc = functools.partial(get_sctm_score, folded_dirname=Path(args.folded))
    pool = mp.Pool(mp.cpu_count())
    sctm_scores_raw_and_ref = list(
        pool.map(pfunc, orig_predicted_backbones, chunksize=5)
    )
    pool.close()
    pool.join()

    sctm_non_nan_idx = [
        i for i, (val, _) in enumerate(sctm_scores_raw_and_ref) if ~np.isnan(val)
    ]
    sctm_scores_mapping = {
        orig_predicted_backbone_names[i]: sctm_scores_raw_and_ref[i][0]
        for i in sctm_non_nan_idx
    }
    sctm_scores_reference = {
        orig_predicted_backbone_names[i]: sctm_scores_raw_and_ref[i][1]
        for i in sctm_non_nan_idx
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

    # For each length category report the number that exceed cutoff
    for length_cat in sctm_scores_with_len["length"].unique():
        passing_num = np.sum(
            (sctm_scores_with_len["scTM"] >= 0.5)
            & (sctm_scores_with_len["length"] == length_cat)
        )
        denom = np.sum(sctm_scores_with_len["length"] == length_cat)
        logging.info(f"{length_cat}: {passing_num}/{denom} passing 0.5 cutoff")

    fig, ax = plt.subplots(dpi=300)
    sns.histplot(sctm_scores_with_len, x="scTM", hue="length")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(
        f"scTM scores, {len(sctm_scores)} generated protein backbones", fontsize=14
    )
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Self-consistency TM score (scTM)", fontsize=12)
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
            [
                (
                    k,
                    sctm_scores_mapping[k],
                    training_tm_scores[k],
                    orig_predicted_backbone_lens[k],
                    orig_predicted_secondary_structs[k][0],
                    orig_predicted_secondary_structs[k][1],
                    sctm_scores_reference[k],
                )
                for k in shared_keys
            ],
            columns=[
                "id",
                "scTM",
                "max training TM",
                "length_int",
                "alpha_counts",
                "beta_counts",
                "scTM best match",
            ],
        )

        # Optionally add in the
        training_tm_scores_match_fname = os.path.join(
            args.predicted, "tm_scores_ref.json"
        )
        if os.path.isfile(training_tm_scores_match_fname):
            with open(training_tm_scores_match_fname) as source:
                tm_matches = json.load(source)
            scores_df["max training TM structure"] = [
                tm_matches[k] for k in shared_keys
            ]

        scores_df["length"] = [
            r"short ($\leq 70$ aa)" if l <= 70 else r"long ($> 70$ aa)"
            for l in scores_df["length_int"]
        ]
        scores_df["designable"] = scores_df["scTM"] >= 0.5

        for l_cat in scores_df["length"].unique():
            subset = scores_df.loc[scores_df["length"] == l_cat]
            sctm_prop = np.mean(subset["scTM"] >= 0.5)
            logging.info(
                f"For {l_cat}, {np.sum(subset['scTM'] >= 0.5)}/{len(subset)}={sctm_prop:.4f} pass 0.5 cutoff"
            )
            designable = subset.loc[subset["designable"]]
            beta_prop = np.mean(designable["beta_counts"] > 0)
            logging.info(
                f"For DESIGNABLE {l_cat}, {np.sum(designable['beta_counts'] > 0)}/{len(designable)}={beta_prop:.4f} with beta sheets"
            )

        for is_designable in scores_df["designable"].unique():
            subset = scores_df.iloc[np.where(scores_df["designable"] == is_designable)]
            beta_prop = np.mean(subset["beta_counts"] > 0)
            beta_count = np.sum(subset["beta_counts"] > 0)
            logging.info(
                f"Designable={is_designable}: beta sheets in {beta_count}/{len(subset)}={beta_prop}"
            )

        scores_df.to_csv(args.outprefix + "_tm_scores.csv")

        r, p = stats.spearmanr(
            scores_df["max training TM"], scores_df["scTM"], alternative="two-sided"
        )
        logging.info(
            f"Spearman's correlation between training TM and scTM: {r:.4g} {p:.4g}"
        )

        # Default figure size is 6.4x4.8
        fig, ax = plt.subplots()
        sns.scatterplot(
            scores_df, x="max training TM", y="scTM", hue="length", alpha=0.5, ax=ax
        )
        ax.axvline(0.5, color="grey", alpha=0.5, linestyle="--")
        ax.axhline(0.5, color="grey", alpha=0.5, linestyle="--")
        # for ax in (jointgrid.ax_joint, jointgrid.ax_marg_x):
        #     ax.axvline(0.5, color="grey", alpha=0.5, linestyle="--")
        # for ax in (jointgrid.ax_joint, jointgrid.ax_marg_y):
        #     ax.axhline(0.5, color="grey", alpha=0.5, linestyle="--")
        ax.set_title("scTM vs. training similarity", fontsize=14)
        ax.set_xlabel("Maximum training TM score", fontsize=12)
        ax.set_ylabel("scTM score", fontsize=12)
        fig.savefig(args.outprefix + "_training_tm_scatter.pdf", bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
