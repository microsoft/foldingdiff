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
import os
import warnings
import glob
import functools
import logging
import argparse
from typing import List, Optional

from tqdm.auto import tqdm

# Verfies that the environment is set up correctly
import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm
import esm.inverse_folding


from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence, AlphabetError


def get_chain_from_pdb(fname: str) -> str:
    """
    Get the chain from the given PDB file. If multiple chains are present,
    return an empty string
    """
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    source = PDBFile.read(fname)
    if source.get_model_count() > 1:
        return ""

    source_struct = source.get_structure()[0]
    chain_ids = set(source_struct.chain_id)
    assert len(chain_ids) == 1
    return chain_ids.pop()


def write_fa(fname: str, seq: str, seqname: str = "sampled"):
    """Write a fasta file"""
    assert fname.endswith(".fasta")
    with open(fname, "w") as f:
        f.write(f">{seqname}\n")
        for chunk in [seq[i : i + 80] for i in range(0, len(seq), 80)]:
            f.write(chunk + "\n")
    return fname


def generate_residues(
    fpath: str, model, chain_id: str = "", n: int = 10, temperature: float = 1.0
) -> List[str]:
    """Generate residues for the structure contained in the PDB file"""
    if not chain_id:
        chain_id = get_chain_from_pdb(fpath)

    structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
    # Coords have shape (seq_len, 3, 3)
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(
        structure
    )
    logging.debug(f"Native sequence: {native_seq}")
    retval = []
    while len(retval) < n:
        # Sample is defined here:
        # https://github.com/facebookresearch/esm/blob/dc0e039dce52ff11e8eadaa1ef96f0cefcc505e9/esm/inverse_folding/gvp_transformer.py
        sampled_seq = model.sample(coords, temperature=temperature)
        try:
            _ = ProteinSequence(sampled_seq)  # Checks alphabet
            logging.debug(f"Sampled sequence: {sampled_seq}")
            retval.append(sampled_seq)
        except AlphabetError:
            # Error; do not return and generate another one
            pass
    return retval


def update_fname(fname: str, i: int, new_dir: str = "") -> str:
    """
    Update the pdb filename to include a numeric index and a .fasta extension.
    If new_dir is given then we move the output filename to that directory.
    """
    assert os.path.isfile(fname)
    parent, child = os.path.split(fname)
    assert child
    child_base, _child_ext = os.path.splitext(child)
    assert child_base
    if new_dir:
        assert os.path.isdir(new_dir), f"Expected {new_dir} to be a directory"
        parent = new_dir
    return os.path.join(parent, f"{child_base}_esm_residues_{i}.fasta")


def build_parser() -> argparse.ArgumentParser:
    """Build a basic CLI"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "fname",
        type=str,
        help="PDB file to generate residues for, or a folder containing these",
    )
    parser.add_argument("-o", "--outdir", type=str, default="", help="Output directory")
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to sample at. Lower values result in lower diversity but higher sequence recovery",
    )
    parser.add_argument(
        "-n", type=int, default=10, help="Number of sequences to generate per structure"
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

    # Load the model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    torch.manual_seed(args.seed)

    pfunc = functools.partial(
        generate_residues, model=model, n=args.n, temperature=args.temperature
    )

    # Load in the file
    if os.path.isfile(args.fname):
        # If output file is given, write it
        sequences = pfunc(args.fname)
        for i, seq in enumerate(sequences):
            out_fname = update_fname(args.fname, i)
            seq_name = os.path.splitext(os.path.basename(out_fname))[0]
            write_fa(out_fname, seq, seqname=seq_name)
    elif os.path.isdir(args.fname):
        # create a subdirecotry to store the fastas
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = os.path.join(args.fname, "esm_generated_fastas")
        os.makedirs(outdir, exist_ok=True)
        logging.info(f"Writing output to {os.path.abspath(outdir)}")
        # Query for inputs and process them
        inputs = sorted(glob.glob(os.path.join(args.fname, "*.pdb")))
        logging.info(f"Found {len(inputs)} PDB files to process in {args.fname}")
        generated = (pfunc(f) for f in tqdm(inputs))
        # Write outputs
        for orig_fname, seqs in zip(inputs, generated):
            for i, seq in enumerate(seqs):
                out_fname = update_fname(orig_fname, i, new_dir=outdir)
                write_fa(
                    out_fname,
                    seq,
                    seqname=os.path.splitext(os.path.basename(out_fname))[0],
                )
    else:
        raise RuntimeError(f"Expected {args.fname} to be a file or directory")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
