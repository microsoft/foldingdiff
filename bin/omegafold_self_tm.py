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
from matplotlib import pyplot as plt

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
        "--outjson",
        type=str,
        default=os.path.join(os.getcwd(), "sctm_scores.json"),
        help="Output json of scores to write",
    )
    parser.add_argument(
        "-P",
        "--plotfile",
        type=str,
        default=os.path.join(os.getcwd(), "sctm_scores.pdf"),
        help="Plot to write histogram of scTM scores to",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    assert os.path.isdir(args.folded)
    assert os.path.isdir(args.predicted)

    orig_predicted_backbones = glob(os.path.join(args.predicted, "generated_*.pdb"))
    logging.info(
        f"Computing selfTM scores across {len(orig_predicted_backbones)} generated structures"
    )
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
    with open(args.outjson, "w") as sink:
        json.dump(sctm_scores_mapping, sink, indent=4)

    fig, ax = plt.subplots()
    ax.hist(sctm_scores, bins=25, alpha=0.6)
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set(
        xlabel=f"scTM, $n={passing_num}$ are designable $(\geq 0.5$)",
        title=f"Self-consistency TM (scTM) scores, {len(sctm_scores)} generated protein backbones",
    )
    fig.savefig(args.plotfile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
