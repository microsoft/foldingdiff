"""
Run MDS on structures to create an embedding visualization

Coloring options:
* training TM similarity
* scTM
* helix/beta strand annotations
* length
"""

import os
import json
import logging
from glob import glob
import argparse


import pandas as pd
from sklearn.manifold import MDS
import umap
from matplotlib import pyplot as plt

from hclust_structures import get_pairwise_tmscores, int_getter
from annot_secondary_structures import count_structures_in_pdb

# :)
SEED = int(
    float.fromhex("2254616977616e2069732061206672656520636f756e74727922") % 10000
)


def len_pdb_structure(fname: str) -> int:
    """Return the integer length of the PDB structure"""
    with open(fname) as source:
        atom_lines = [l.strip() for l in source if l.startswith("ATOM")]
    last_line_tokens = atom_lines[-1].split()
    last_line_l = int(last_line_tokens[5])
    assert int(len(atom_lines) / 3) == last_line_l
    return last_line_l


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mdsdirname", type=str, help="Directory containing PDB files")
    group.add_argument("--gitscores", type=str, default="", help="Git scores json")
    parser.add_argument("--sctm", type=str, default="", help="scTM scores JSON file")
    parser.add_argument(
        "--trainingtm", type=str, default="", help="Training TM score JSON"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tmscore_mds",
        help="PDF file prefix to write output to",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    if args.mdsdirname:
        # Get files
        fnames = sorted(
            glob(os.path.join(args.mdsdirname, "*.pdb")),
            key=lambda x: int_getter(os.path.basename(x)),
        )
        logging.info(f"Computing TMscore on {len(fnames)} structures")

        # in the dissimilarity matrix, larger values should indicate more distant points
        # therefore we want to do 1 - TMscore (since larger values are closer in TMscore space)
        pdist_df = 1.0 - get_pairwise_tmscores(fnames, sctm_scores_json=args.sctm)
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=False,  # TMscores do not respect triangle inequality
            n_jobs=-1,
            random_state=SEED,
        )
        embedding = pd.DataFrame(
            mds.fit_transform(pdist_df.values),
            index=pdist_df.index,
            columns=["MDS1", "MDS2"],
        )
    elif args.gitscores:
        git_df = pd.read_csv(args.gitscores, index_col=0, sep=" ", header=None)
        fnames = [os.path.abspath(f) for f in git_df.index]
        git_df.index = [os.path.basename(f).split(".")[0] for f in git_df.index]
        # Remove columsn of all nan
        git_df.dropna(axis=1, how="all", inplace=True)
        embedding = pd.DataFrame(
            umap.UMAP(random_state=SEED).fit_transform(git_df.values),
            index=git_df.index,
            columns=["UMAP1", "UMAP2"],
        )
    else:
        raise ValueError("Must specify either --mdsdirname or --gitscores")

    format_strings = {
        "Number helices": "{x:.1f}",
    }
    # For a variety of coloring keys, compute/read the scores and color scatter
    # plot by the scores.
    for k, v in {
        "Max training TM": args.trainingtm,
        "scTM": args.sctm,
        "length": len_pdb_structure,
        "Number helices": lambda x: count_structures_in_pdb(x, "psea")[0],
        "Number sheets": lambda x: count_structures_in_pdb(x, "psea")[1],
        "null": None,
    }.items():
        if v is None or v:
            logging.info(f"Coloring by {k} scores")
            figsize = (6.4, 4.8)
            annot_points = False
            if v is None:
                scores = None
                figsize = (12.8, 9.6)
                annot_points = True
                # If we are doing the null, the plot very big and label each
                # point with the text id
            elif callable(v):
                fname_to_key = lambda f: os.path.basename(f).split(".")[0]
                scores = {
                    fname_to_key(f): v(f)
                    for f in fnames
                    if fname_to_key(f) in embedding.index
                }
                scores = embedding.index.map(scores)
            elif os.path.isfile(v):
                with open(v) as source:
                    scores = embedding.index.map(json.load(source))
            else:
                raise ValueError(f"Invalid value for {k}: {v}")

            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            points = ax.scatter(
                embedding.iloc[:, 0],
                embedding.iloc[:, 1],
                s=8,
                c=scores,
                cmap="RdYlBu",
                alpha=0.9,
            )
            if annot_points:
                for i in range(len(embedding)):
                    ax.annotate(
                        embedding.index[i],
                        (embedding.iloc[i, 0], embedding.iloc[i, 1]),
                        fontsize=6,
                    )
            ax.set(
                xlabel=embedding.columns[0],
                ylabel=embedding.columns[1],
            )
            if not k == "null":
                ax.set(
                    xticks=[],
                    yticks=[],
                    title=k,
                )
            if scores is not None:
                cbar = plt.colorbar(
                    points,
                    ax=ax,
                    fraction=0.08,
                    pad=0.04,
                    location="right",
                    # format=format_strings.get(k, None),
                )
                cbar.ax.set_ylabel(k, fontsize=12)

            fig.savefig(f"{args.output}_mds_{k}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
