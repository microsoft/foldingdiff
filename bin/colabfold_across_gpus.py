"""
Short script to parallelize colabfold to run across GPUs to speed up runtime. Make
sure to set export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 before running!
"""

# Use only standard libraries so we don't need to modify the env
import functools
import os
import glob
import logging
import argparse
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import *

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """
    Build a basic CLI
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "foldername", type=str, help="Folder containing a3m msa files to run"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.abspath(os.path.join(os.getcwd(), "colabfold_predictions")),
        help="Output directory",
    )
    parser.add_argument(
        "-g", "--gpus", type=int, nargs="*", default=[0, 1, 2, 3], help="GPUs to use"
    )
    return parser


def run_colabfold(input_a3m: Path, outdir: Path, gpu: int) -> None:
    """Run colabfold on the given a3m MSA file"""
    # Example command: colabfold_batch msas predictions --use-gpu-relax --amber --num-recycle=3 --model-type=AlphaFold2-ptm
    executable = shutil.which("colabfold_batch")
    if not executable:
        raise FileNotFoundError("Could not find colabfold_batch in PATH")
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {executable} {input_a3m} {outdir} --use-gpu-relax --amber --num-recycle=3 --model-type=AlphaFold2-ptm"
    retval = subprocess.call(cmd, shell=True)
    assert (
        retval == 0
    ), f"colabfold_batch on {input_a3m} failed with return code {retval}"


def run_colabfold_multi(input_a3m_files: List[Path], gpu: int, outdir: Path) -> None:
    for f in input_a3m_files:
        run_colabfold(f, outdir, gpu)


def main():
    """
    Run the script
    """
    parser = build_parser()
    args = parser.parse_args()

    assert "XLA_FLAGS" in os.environ, "XLA_FLAGS not set!"

    # Create output directory
    outdir = Path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Get all a3m files in the input directory
    input_files = glob.glob(os.path.join(args.foldername, "*.a3m"))[:4]
    assert input_files, f"No a3m files found in {args.foldername}"

    # Split the input_files into chunks equal to the number of GPUs
    indices = np.array_split(np.arange(len(input_files)), len(args.gpus))
    input_files_split = []
    for idx in indices:
        input_files_split.append([input_files[i] for i in idx])
    assert len(input_files_split) == len(args.gpus)
    logging.info(
        f"Splitting input {len(input_files)} into sizes {[len(i) for i in indices]}"
    )

    pfunc = functools.partial(run_colabfold_multi, outdir=outdir)
    # Create processes for each set of files
    processes = [
        mp.Process(target=pfunc, args=(input_files_split[i], args.gpus[i]))
        for i in range(len(args.gpus))
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
