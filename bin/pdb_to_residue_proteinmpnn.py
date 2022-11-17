import os
from glob import glob
import logging
import argparse
import subprocess
from pathlib import Path
import tempfile
from typing import *

from tqdm.auto import tqdm

def write_fasta(fname: str, seq: str, seqname: str = "sampled") -> str:
    """Write a fasta file"""
    assert fname.endswith(".fasta")
    with open(fname, "w") as f:
        f.write(f">{seqname}\n")
        for chunk in [seq[i : i + 80] for i in range(0, len(seq), 80)]:
            f.write(chunk + "\n")
    return fname

def read_fasta(fname: str) -> Dict[str, str]:
    """
    Read the given fasta file and return a dictionary of its sequences
    """
    seq_dict = {}
    curr_key, curr_seq = "", ""
    with open(fname, "r") as source:
        for line in source:
            if line.startswith(">"):
                if curr_key:
                    assert curr_seq
                    seq_dict[curr_key] = curr_seq
                curr_key = line.strip().strip(">")
                curr_seq = ""
            else:
                curr_seq += line.strip()

        assert curr_key and curr_seq
        seq_dict[curr_key] = curr_seq
    return seq_dict


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
    return os.path.join(parent, f"{child_base}_proteinmpnn_residues_{i}.fasta")


def generate_residues_proteinmpnn(
    pdb_fname: str, n_sequences: int = 8, temperature: float = 0.1
) -> List[str]:
    """
    Generates residues for the given pdb_filename using ProteinMPNN

    Trippe et al uses a temperature of 0.1 to sample 8 amino acid sequences per structure
    """
    bname = os.path.basename(pdb_fname).replace(".pdb", ".fa")
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        cmd = f'python ~/software/ProteinMPNN/protein_mpnn_run.py --pdb_path_chains A --out_folder {tempdir} --num_seq_per_target {n_sequences} --seed 1234 --batch_size {n_sequences} --pdb_path {pdb_fname} --sampling_temp "{temperature}" --ca_only'
        retval = subprocess.call(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        assert retval == 0, f"Command {cmd} failed with return value {retval}"
        outfile = tempdir / "seqs" / bname
        assert os.path.isfile(outfile)

        # Read the fasta file, return the sequences that were generated
        seqs = read_fasta(outfile)
        seqs = {k: v for k, v in seqs.items() if k.startswith("T=")}
    assert len(seqs) == n_sequences
    return list(seqs.values())


def build_parser():
    """Build a CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dirname",
        type=str,
        help="Folder containing PDB files to run ProteinMPNN on",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.join(os.getcwd(), "proteinmpnn_residues"),
        help="Output directory for fasta files",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature to sample at",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=8,
        help="Number of sequences to generate per structure",
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=False)
    logging.info(f"Writing outputs to {args.outdir}")

    pdb_fnames = list(glob(os.path.join(args.dirname, "*.pdb")))
    logging.info(f"Running ProteinMPNN on {len(pdb_fnames)} PDB files")

    for pdb_fname in tqdm(pdb_fnames):
        seqs = generate_residues_proteinmpnn(pdb_fname, n_sequences=args.num, temperature=args.temperature)
        for i, seq in enumerate(seqs):
            out_fname = update_fname(pdb_fname, i, new_dir=args.outdir)
            write_fasta(out_fname, seq, seqname=os.path.splitext(os.path.basename(out_fname))[0])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
