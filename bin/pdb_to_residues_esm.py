"""
Short script to convert a PDB file with coordinates to a list of residues.
This is coutesy of ESM,

Please note that this is _NOT_ compatible with the protein diffusion main conda environment and must
use a different environment. To set up this environment, do the following:

mamba create -n inverse -c conda-forge pytorch_geometric pytorch-gpu=1.10
conda activate inverse
pip install biotite
pip install git+https://github.com/facebookresearch/esm.git
"""

# uses the following notebook as a reference:
# https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb
import logging
import argparse

# Verfies that the environment is set up correctly
import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm
import esm.inverse_folding


def write_fa(fname: str, seq: str):
    """Write a fasta file"""
    with open(fname, "w") as f:
        f.write(">sampled\n")
        for chunk in [seq[i : i + 80] for i in range(0, len(seq), 80)]:
            f.write(chunk + "\n")
    return fname


def build_parser() -> argparse.ArgumentParser:
    """Build a basic CLI"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fname", type=str, help="PDB file to generate residues for")
    parser.add_argument(
        "-c", "--chain", type=str, default="A", help="Chain to use within PDB file"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to sample at. Lower values result in lower diversity but higher sequence recovery",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output file (fasta format) to write to. If not provided, default to input + .fasta",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=int(float.fromhex("4675636b20796f75205869204a696e70696e67") % 10000),
        help="Random seed",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load in the file
    fpath = args.fname
    chain_id = args.chain
    structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
    # Coords have shape (seq_len, 3, 3)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(
        structure
    )
    logging.info(f"Native sequence: {native_seq}")

    # Load the model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    # Sample the residues
    torch.manual_seed(args.seed)
    sampled_seq = model.sample(coords, temperature=1.0)
    logging.info(f"Sampled sequence: {sampled_seq}")

    # If output file is given, write it
    out = args.output
    if out == "":
        out = fpath.replace(".pdb", ".fasta")
    if args.output:
        write_fa(out, sampled_seq)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
