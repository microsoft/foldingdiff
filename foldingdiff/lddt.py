"""
Code for computing lDDT scores.

Usage as a script to calculate the lDDT betwen each sampled structure and its
corresponding folded structures as used for scTM calculation:
python lddt.py <sampled_dir> <folded_dir>

Writes a json file with lDDT scores for each sampled structure to its correpsonding
folded structures
"""

import os, sys
from collections import defaultdict
import logging
from pathlib import Path
import subprocess
import shutil
import multiprocessing as mp
import tempfile
import json

import pandas as pd

from tqdm.auto import tqdm

IMAGE = "2d07309e7a56"  # Docker image from https://git.scicore.unibas.ch/schwede/openstructure/container_registry/

DOCKER_OST = Path(os.path.realpath(__file__)).parent.parent / "scripts/run_docker_ost"
assert DOCKER_OST.exists(), f"Cannot find docker wrapper script {DOCKER_OST}"


def lddt(query: Path, ref: Path) -> float:
    """Compute the lDDT between query and reference structures."""
    assert query.exists(), f"Cannot find query structure {query}"
    assert ref.exists(), f"Cannot find reference structure {ref}"

    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(query, tmpdir)
        shutil.copy(ref, tmpdir)
        os.chdir(tmpdir)

        cmd = f"{DOCKER_OST} {IMAGE} compare-structures -m {os.path.basename(str(query))} -r {os.path.basename(str(ref))} --lddt -o lddt.json"
        subprocess.call(cmd, shell=True)

        if not os.path.exists("lddt.json"):
            logging.error(f"Failed to compute lDDT for {query} and {ref}")
            return -1.0

        with open("lddt.json", "r") as outfile:
            data = json.load(outfile)

    os.chdir(orig_dir)  # Return to original directory
    if "lddt" in data:
        return data["lddt"]
    return -1.0


def lddt_sampled_folded(sampled_dir: Path, folded_dir: Path):
    """
    For each sampled structure, compute the lDDT to each of its corresponding
    folded structures
    """
    sampled_pdbs = sorted(list(sampled_dir.glob("*.pdb")))
    logging.info(f"Found {len(sampled_pdbs)} sampled structures in {sampled_dir}")

    sampled_to_folded_pdbs = {
        s: list(folded_dir.glob(f"{s.stem}_*.pdb")) for s in sampled_pdbs
    }
    n_matches = [len(v) for v in sampled_to_folded_pdbs.values()]
    logging.info(
        f"Found {sum(n_matches) / len(n_matches)} matching folded structures per sampled structure in {folded_dir}"
    )

    # Flatten the dictionary
    sampled_folded_pairs = []
    for sampled_pdb, folded_pdbs in sampled_to_folded_pdbs.items():
        for folded_pdb in folded_pdbs:
            # Ordering is query -> ref for the lddt function call later under starmap
            sampled_folded_pairs.append((folded_pdb, sampled_pdb))

    pool = mp.Pool(int(mp.cpu_count() // 2))
    lddt_values = pool.starmap(
        lddt,
        sampled_folded_pairs,
        chunksize=10,
    )
    pool.close()
    pool.join()

    # Compute lDDT for each sampled structure
    out_dict = defaultdict(dict)
    for (folded_pdb, sampled_pdb), l_val in zip(sampled_folded_pairs, lddt_values):
        out_dict[str(sampled_pdb.stem)][str(folded_pdb.stem)] = l_val

    # Write out the results
    out_path = "lddt.json"
    logging.info(f"Writing lDDT scores to {out_path}")
    with open(out_path, "w") as sink:
        json.dump(out_dict, sink, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # print(lddt(Path(sys.argv[1]), Path(sys.argv[2])))
    lddt_sampled_folded(Path(sys.argv[1]), Path(sys.argv[2]))
