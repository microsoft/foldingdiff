"""
Wrapper to load in a PDB file as a pyrosetta pose, relax it, and write back
out the relaxed pose as a PDB file.
"""
import os
import glob
import multiprocessing as mp
import logging
import argparse

import pyrosetta
from pyrosetta import rosetta

pyrosetta.init()
logging.basicConfig(level=logging.INFO)


def relax_pdb(fname: str, out_fname: str) -> str:
    """Relax the pose."""
    pose = rosetta.core.import_pose.pose_from_file(fname)
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(pyrosetta.get_fa_scorefxn())
    relax.apply(pose)
    pose.dump_pdb(out_fname)
    return out_fname


def build_parser() -> argparse.ArgumentParser:
    """Basic CLI parser."""
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputdir", type=str, help="Input dir of .pdb files")
    parser.add_argument("outdir", type=str, help="Output dir for relaxed .pdb files")
    return parser


def main():
    """Run script."""
    args = build_parser().parse_args()

    # Get the pdb files in the directory
    pdb_files = glob.glob(os.path.join(args.inputdir, "*.pdb"))
    logging.info(f"Found {len(pdb_files)} pdb files in {args.inputdir}")

    # Make the output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Run each through relaxation
    out_fnames = [
        os.path.join(args.outdir, os.path.basename(fname)) for fname in pdb_files
    ]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(relax_pdb, zip(pdb_files, out_fnames), chunksize=5)


if __name__ == "__main__":
    main()
