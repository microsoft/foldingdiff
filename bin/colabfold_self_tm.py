"""
Script for calculating the self tm score given the input as sampled structures
and the structures resulting from running the structures via inverse folding
residue generation and alphafold/colabfold. Expects the following directory
structure:
working_dir (where this script is run):
    - sampled_pdb (containing the original generated pdb structures)
    - msas (containing the a3m files from MSA generation)
    - colabfold_predictions (containing the results from colabfold)
    
"""

import os, sys
import logging
import glob
import json
import argparse
from pathlib import Path
from typing import *

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import tmalign


def get_sctm_score(orig_pdb: Path, folded_pdb_dirs: List[Path]) -> float:
    """
    Get the scTM score given the original pdb file and list of dirs with folded pdbs
    """
    if not folded_pdb_dirs:
        return 0.0
    folded_pdb_files = []
    for d in folded_pdb_dirs:
        matches = glob.glob(str(d / "*_relaxed_rank_*_model_*.pdb"))
        assert len(matches) <= 5
        folded_pdb_files.extend(matches)
    if not folded_pdb_files:
        return 0.0
    logging.debug(
        f"Matching {orig_pdb} against {len(folded_pdb_files)} folded structures"
    )

    # Get the scTM score
    score = tmalign.max_tm_across_refs(orig_pdb, folded_pdb_files, chunksize=1)
    return score


def seqname_from_a3m(a3m_path: str) -> str:
    """
    Gets the original query sequence from a3m
    """
    # Return the first header line
    with open(a3m_path) as source:
        for line in source:
            if line.startswith(">"):
                return line.split()[0][1:].strip()


def build_parser() -> argparse.ArgumentParser:
    """Get the CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getcwd(),
        help="Directory containing the results",
    )
    return parser


def main() -> None:
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    assert os.path.isdir(results_dir)

    # Get the list of generated structures
    gen_struct_subdir = results_dir / "sampled_pdb"
    assert os.path.isdir(gen_struct_subdir)
    gen_structs = glob.glob(str(gen_struct_subdir / "*.pdb"))
    assert gen_structs
    # Dictionary mapping names to path to structure, something like generated_123
    gen_structs = {os.path.splitext(os.path.basename(s))[0]: s for s in gen_structs}

    # Get the list of inverse folding sequences
    msa_subdir = results_dir / "msas"
    msa_files = glob.glob(str(msa_subdir / "*.a3m"))
    assert msa_files
    # Create a mapping from the non-readable auto-generated MSA names to the readable normal names
    msa_name_to_human_name = {
        os.path.splitext(os.path.basename(s))[0]: seqname_from_a3m(s) for s in msa_files
    }

    # Query the list of folded structures. This should contain folders corresponding to the
    # cryptic msa names
    fold_subdir = results_dir / "colabfold_predictions"
    fold_subdir_contents = [fold_subdir / d for d in os.listdir(fold_subdir)]
    fold_subdir_contents = [d for d in fold_subdir_contents if os.path.isdir(d)]
    # Walk through and map each to a generated structure name gen_struct_names
    gen_struct_to_folded_structs = {s: [] for s in gen_structs}
    for d in fold_subdir_contents:
        d_readable = msa_name_to_human_name[os.path.basename(d)]
        generated_base = "_".join(d_readable.split("_")[:2])
        assert generated_base in gen_structs
        gen_struct_to_folded_structs[generated_base].append(d)

    # For each set of reference and generated structures, compute score
    sctm_scores = {}
    for s in tqdm(gen_structs):
        sctm_scores[s] = get_sctm_score(gen_structs[s], gen_struct_to_folded_structs[s])

    # Write output
    with open("sctm_scores.json", "w") as sink:
        json.dump(sctm_scores, sink, indent=2)

    fig, ax = plt.subplots(dpi=300)
    ax.hist(sctm_scores.values(), bins=20)
    ax.set(
        xlabel="scTM score",
        title=f"Self-consistency TM scores for {len(sctm_scores)} generated structures",
    )
    fig.savefig("sctm_scores.pdf", bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
