"""
Short and easy wrapper for TMalign
"""

import os
import argparse
import collections
import re
import itertools
import shutil
import subprocess
import multiprocessing as mp
import logging
from typing import *

import numpy as np

logging.basicConfig(level=logging.INFO)


def run_tmalign(query: str, reference: str, fast: bool = False) -> float:
    """
    Run TMalign on the two given input pdb files
    """
    assert os.path.isfile(query)
    assert os.path.isfile(reference)

    # Check if TMalign is installed
    exec = shutil.which("TMalign")
    if not exec:
        raise FileNotFoundError("TMalign not found in PATH")

    # Build the command
    cmd = f"{exec} {query} {reference}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"Tmalign failed on {query}|{reference}, returning NaN")
        return np.nan

    # Parse the outpu
    score_lines = []
    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            score_lines.append(line)

    # Fetch the chain number
    key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    return results_dict["Chain_2"]  # Normalize by reference length


def max_tm_across_refs(
    query: str,
    references: List[str],
    n_threads: int = mp.cpu_count(),
    fast: bool = True,
    chunksize: int = 10,
    parallel: bool = True,
) -> Tuple[float, str]:
    """
    Compare the query against each of the references in parallel and return the maximum score
    along with the corresponding reference
    This is typically a lot of comparisons so we run with fast set to True by default
    """
    logging.debug(
        f"Matching against {len(references)} references using {n_threads} workers with fast={fast}"
    )
    args = [(query, ref, fast) for ref in references]
    if parallel and len(references) > 1:
        n_threads = min(n_threads, len(references))
        pool = mp.Pool(n_threads)
        values = list(pool.starmap(run_tmalign, args, chunksize=chunksize))
        pool.close()
        pool.join()
    else:
        values = list(itertools.starmap(run_tmalign, args))

    return np.nanmax(values), references[np.argmax(values)]


def match_files(
    query_files: Collection[str],
    ref_files: Collection[str],
    strategy: str = "exact",
) -> Dict[str, List[str]]:
    """Match the files."""
    query_files_map = {os.path.splitext(os.path.basename(f))[0]: f for f in query_files}
    ref_files_map = {os.path.splitext(os.path.basename(f))[0]: f for f in ref_files}

    retval = collections.defaultdict(list)
    if strategy == "exact":
        for k in query_files_map:
            if k in ref_files_map:
                retval[query_files_map[k]].append(ref_files_map[k])
    elif strategy == "prefix":
        for k in query_files_map:
            pattern = re.compile("^" + k + r"[\-\_]+.*")
            for k2 in [k2 for k2 in ref_files_map if pattern.match(k2)]:
                retval[query_files_map[k]].append(ref_files_map[k2])
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return retval


def parse_args() -> argparse.Namespace:
    """Basic CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, nargs="+", help="Query files")
    parser.add_argument("-r", "--ref", type=str, nargs="+", help="Reference files")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument(
        "-s",
        "--strat",
        type=str,
        choices=["exact", "prefix"],
        default="exact",
        help="Strategy for matching query and reference files",
    )
    return parser.parse_args()


def main():
    """Run as a script."""
    args = parse_args()

    query2refs = match_files(args.query, args.ref, args.strat)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        out = list(pool.starmap(max_tm_across_refs, query2refs.items()))
        tmscores, _best_matching = zip(*out)

    logging.info(f"Mean TM-score: {np.nanmean(tmscores):.3f}")
    logging.info(f"Num >= 0.5: {np.sum(np.array(tmscores) >= 0.5)} / {len(tmscores)}")


if __name__ == "__main__":
    main()
