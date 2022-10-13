"""
Run heirarchical clustering on the pairwise distance matrix between all pairs of files
"""

import os, sys
import re
import json
import logging
from glob import glob
from pathlib import Path
import itertools
import argparse
from typing import *
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns

from train import get_train_valid_test_sets

from foldingdiff import tmalign

# :)
SEED = int(float.fromhex("2254616977616e2069732061206672656520636f756e74727922") % 10000)

def int_getter(x: str) -> int:
    """Fetches integer value out of a string"""
    matches = re.findall(r"[0-9]+", x)
    assert len(matches) == 1
    return int(matches.pop())


def get_pairwise_tmscores(
    fnames: Collection[str], sctm_scores_json: Optional[str] = None
) -> pd.DataFrame:
    """Get the pairwise TM scores across all fnames"""
    logging.info(f"Computing pairwise distances between {len(fnames)} pdb files")

    bname_getter = lambda x: os.path.splitext(os.path.basename(x))[0]
    if sctm_scores_json:
        with open(sctm_scores_json) as source:
            sctm_scores = json.load(source)
        fnames = [f for f in fnames if sctm_scores[bname_getter(f)] >= 0.5]
        logging.info(f"{len(fnames)} structures have scTM scores >= 0.5")

    # for debugging
    # fnames = fnames[:50]

    pairs = list(itertools.combinations(fnames, 2))
    pool = mp.Pool(mp.cpu_count())
    values = list(pool.starmap(tmalign.run_tmalign, pairs, chunksize=25))
    pool.close()
    pool.join()

    bnames = [bname_getter(f) for f in fnames]
    retval = pd.DataFrame(1.0, index=bnames, columns=bnames)
    for (k, v), val in zip(pairs, values):
        retval.loc[bname_getter(k), bname_getter(v)] = val
        retval.loc[bname_getter(v), bname_getter(k)] = val
    assert np.allclose(retval, retval.T)
    return retval


def build_parser():
    """Build a CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dirname", type=str, help="Directory of PDB files to analyze")
    g.add_argument("--testsubset", type=int, help="Subset of test set sequences to run")
    parser.add_argument(
        "--sctm", type=str, required=False, default="", help="scTM scores to filter by"
    )
    parser.add_argument("-o", "--output", type=str, default="tmscore_hclust.pdf", help="PDF file to write output clustering plot")
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    # Get the files
    if args.dirname:
        fnames = sorted(
            glob(os.path.join(args.dirname, "*.pdb")),
            key=lambda x: int_getter(os.path.basename(x)),
        )
        assert fnames, f"{args.dirname} does not contain any pdb files"
    elif args.testsubset:
        # We only care about fnames here
        *_, test_subset = get_train_valid_test_sets(
            max_seq_len=128,
            min_seq_len=50,
            seq_trim_strategy="discard",
        )
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(test_subset.filenames), size=args.testsubset, replace=False)
        fnames = [test_subset.filenames[i] for i in idx]
    else:
        raise NotImplementedError

    # TMscore of 1 = perfect match --> 0 distance, so need 1.0 - tmscore
    pdist_df = 1.0 - get_pairwise_tmscores(fnames, sctm_scores_json=args.sctm)

    # https://stackoverflow.com/questions/38705359/how-to-give-sns-clustermap-a-precomputed-distance-matrix
    # https://stackoverflow.com/questions/57308725/pass-distance-matrix-to-seaborn-clustermap
    m = "average"  # Trippe uses average here
    linkage = hc.linkage(
        sp.distance.squareform(pdist_df), method=m, optimal_ordering=False
    )

    c = sns.clustermap(
        pdist_df,
        row_linkage=linkage,
        col_linkage=linkage,
        method=None,
        row_cluster=True,
        col_cluster=True,
        vmin=0.0,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": r"$d(x, y) = 1 - \mathrm{TMscore}(x, y)$"},
    )
    c.savefig(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
