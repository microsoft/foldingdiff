"""
Code to visualize with pymol
"""
import os
import argparse
import tempfile
from typing import *

import imageio
import pymol

from tqdm.auto import tqdm


def pdb2png(pdb_fname: str, png_fname: str) -> str:
    """Convert the pdb file into a png, returns output filename"""
    # https://gist.github.com/bougui505/11401240
    pymol.cmd.load(pdb_fname)
    pymol.cmd.show("cartoon")
    pymol.cmd.set("ray_opaque_background", 0)
    pymol.cmd.png(png_fname, ray=1, dpi=600)
    pymol.cmd.delete("*")  # So we dont' draw multiple images at once
    return png_fname


def pdb2png_from_args(args):
    """Wrapper for the above to handle CLI args"""
    pdb2png(args.input, args.output)


def images_to_gif(images: Collection[str], fname: str) -> str:
    """Create a gif from the given images, returns output filename"""
    # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    images_read = [imageio.imread(f) for f in images]
    imageio.mimsave(fname, images_read)
    return fname


def images_to_gif_from_args(args):
    """Wrapper for the above to handle CLI args"""
    with tempfile.TemporaryDirectory() as tempdir:
        image_filenames = []
        for i, fname in tqdm(enumerate(args.input)):
            assert os.path.isfile(fname)
            outname = pdb2png(fname, os.path.join(tempdir, f"pdb_file_{i}.png"))
            image_filenames.append(outname)
        gif = images_to_gif(image_filenames, args.output)


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

    return parser


def main():
    """Run this as an interactive script"""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
