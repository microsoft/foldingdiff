"""
Code to visualize with pymol

Example usage:
python ~/protdiff/protdiff/pymol_vis.py pdb2gif -i projects/generated/generation-with-history/sampled_pdb/sample_history/generated_0/*.pdb -o generated_0.gif
"""
import os
import glob
import re
import multiprocessing as mp
import argparse
import tempfile
from pathlib import Path
from typing import *

import biotite.structure as struc
import biotite.structure.io as strucio
import imageio
import pymol

from tqdm.auto import tqdm


def pdb2png(pdb_fname: str, png_fname: str) -> str:
    """Convert the pdb file into a png, returns output filename"""
    # https://gist.github.com/bougui505/11401240
    pymol.cmd.load(pdb_fname)
    pymol.cmd.show("cartoon")
    pymol.cmd.color("green")
    pymol.cmd.set("ray_opaque_background", 0)
    pymol.cmd.png(png_fname, ray=1, dpi=600)
    pymol.cmd.delete("*")  # So we dont' draw multiple images at once
    return png_fname


def pdb2png_from_args(args):
    """Wrapper for the above to handle CLI args"""
    pdb2png(args.input, args.output)


def pdb2png_dir_from_args(args):
    """Wrapper to call pdb2png in parallel from CLI args"""
    os.makedirs(args.output, exist_ok=True)
    input_fnames = glob.glob(os.path.join(args.input, "*.pdb"))
    arg_tuples = [
        (fname, os.path.join(args.output, os.path.basename(fname)))
        for fname in input_fnames
    ]
    pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(pdb2png, arg_tuples, chunksize=5)
    pool.close()
    pool.join()


def images_to_gif(
    images: Collection[str], fname: str, pause_on_last: bool = True, loop: bool = True
) -> str:
    """Create a gif from the given images, returns output filename"""
    # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    images_read = [imageio.imread(f) for f in images]
    # Show each frame for 0.05 seconds, and the last one for 10 seconds
    frametimes = [0.05] * len(images)
    if pause_on_last:
        frametimes[-1] = 10.0
    imageio.mimsave(
        fname,
        images_read,
        duration=frametimes,
        subrectangles=True,
        loop=0 if loop else 1,  # 0 = loop indefinitely, 1 = loop once
    )
    return fname


def _align_two_pdb_files(query_fname: str, ref_fname: str, output_fname: str) -> str:
    """Align two pdb files, and save the aligned query to the output file"""
    # Uses the superimpose command from biotite
    # https://www.biotite-python.org/apidoc/biotite.structure.superimpose.html
    query_struc = strucio.load_structure(query_fname)
    ref_struc = strucio.load_structure(ref_fname)
    # args are (fixed, mobile)
    fitted, _ = struc.superimpose(ref_struc, query_struc)

    strucio.save_structure(output_fname, fitted)
    return output_fname


def images_to_gif_from_args(args):
    """Wrapper for the above to handle CLI args"""
    get_int_tuple = lambda s: tuple(int(i) for i in re.findall(r"[0-9]+", s))
    sorted_inputs = sorted(args.input, key=get_int_tuple)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        # Superimpose each consecutive pair of images so that the animation
        # is smooth(er). The "reference" image is the last image, so within
        # each pair the first is the query and the second is the reference.
        aligned_pdb_files = [sorted_inputs[-1]]

        for query in sorted_inputs[:-1][::-1]:
            # Reference is the prior aligned pdb fil3e
            aligned = _align_two_pdb_files(
                query,
                aligned_pdb_files[-1],
                os.path.join(tempdir, os.path.basename(query)),
            )
            aligned_pdb_files.append(aligned)
        # From final -> first to first -> final
        aligned_pdb_files = aligned_pdb_files[::-1]

        # Create pairs of (pdb, png) filenames
        arg_tuples = [
            (fname, os.path.join(tempdir, f"pdb_file_{i}.png"))
            for i, fname in enumerate(aligned_pdb_files)
        ]
        with mp.Pool(mp.cpu_count()) as pool:
            image_filenames = list(pool.starmap(pdb2png, arg_tuples, chunksize=5))
        gif = images_to_gif(image_filenames, args.output)
        assert gif


def build_parser():
    """Build a basic CLI parser"""
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(help="Usage modes")

    # For converting a bunch of pdb to a gif
    gif_parser = subparsers.add_parser(
        "pdb2gif", help="Convert a series of input PDB structures into a gif animation"
    )
    gif_parser.add_argument(
        "-i", "--input", type=str, nargs="*", required=True, help="PDB files to consume"
    )
    gif_parser.add_argument(
        "-o", "--output", type=str, required=True, help="Gif file to write"
    )
    gif_parser.set_defaults(func=images_to_gif_from_args)

    # For converting single PDB
    png_parser = subparsers.add_parser(
        "pdb2png", help="Convert a single pdb file to a single png file"
    )
    png_parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input pdb file"
    )
    png_parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output file to write"
    )
    png_parser.set_defaults(func=pdb2png_from_args)

    # For converting an entire batch of PDBs to PNG files
    png_batch_parser = subparsers.add_parser(
        "pdb2png_batch",
        help="Convert a folder containing PDB files to a folder of PNG files",
    )
    png_batch_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Directory containing *.pdb files",
    )
    png_batch_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Directory to write *.png files to",
    )
    png_batch_parser.set_defaults(func=pdb2png_dir_from_args)

    return parser


def main():
    """Run this as an interactive script"""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
