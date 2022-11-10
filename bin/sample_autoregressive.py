"""
Code to sample from an autoregressive transformer model
"""

import os, sys
import logging
from pathlib import Path
import argparse

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch


from foldingdiff import modelling, utils
from foldingdiff import angles_and_coords as ac


def sample_initial_angles(
    n_samples: int, n_angles: int = 4, eps: float = 1e-4, seed=1234
) -> torch.Tensor:
    """
    Sample initial angles based on naturally occurring proteins
    Returned tensor has shape (n_samples, n_angles, 6) for 6 angles
    Of this, the first n_angles are set; the rest are random
    """
    # Sample initial angles
    cwd = Path(__file__).parent
    pdb_files = list((cwd / Path("../data/cath/dompdb")).glob("*"))
    assert pdb_files
    logging.info(f"Sampling from {len(pdb_files)} PDB files")

    retval = torch.zeros((n_samples, n_angles, 6))
    rng = np.random.default_rng(seed)
    for i, idx in enumerate(rng.integers(0, len(pdb_files), n_samples)):
        fname = pdb_files[idx]
        angles = ac.canonical_distances_and_dihedrals(
            fname, angles=ac.EXHAUSTIVE_ANGLES, distances=ac.MINIMAL_DISTS
        )
        assert angles is not None, f"Error when parsing {pdb_files[idx]}"
        retval[i, :, :] = torch.from_numpy(angles.values[:n_angles, :])

    # If given, add a bit of noise to the angles as well
    if eps:
        logging.info(f"Adding noise zero means and variance {eps} to angles")
        torch.manual_seed(seed)
        noise = torch.randn((n_samples, n_angles, 6)) * eps
        retval += noise

    retval = utils.modulo_with_wrapped_range(retval, -np.pi, np.pi)
    return retval


def build_parser():
    """
    Build a basic CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", help="Model name or path", type=str)
    parser.add_argument("-o", "--outdir", help="Output directory", default=".")
    parser.add_argument(
        "--num",
        default=10,
        type=int,
        help="Number of samples at each length from 50-128",
    )
    parser.add_argument(
        "--num_angles",
        help="Number of angles to sample as seed angles",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-l",
        "--lengths",
        type=int,
        nargs=2,
        default=[50, 128],
        help="Range of lengths to sample from",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run generations on",
    )
    return parser


def main() -> None:
    """
    Run the script
    """
    parser = build_parser()
    args = parser.parse_args()

    outdir = Path(args.outdir).absolute()

    logging.info(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)
    assert not os.listdir(outdir), f"Expected {outdir} to be empty"

    # Load the model
    device = torch.device(args.device)
    m = modelling.BertForAutoregressive.from_dir(
        args.model, copy_to=Path(outdir) / "model_snapshot"
    ).to(device)

    # Load the model offsets
    angle_offsets = torch.from_numpy(
        np.load(Path(args.model) / "training_mean_offset.npy")
    ).to(device)

    # Sample initial angles; 10 initializations, 4 angles
    initial_angles = sample_initial_angles(args.num, args.num_angles, eps=0.0).to(
        device
    )
    # Shift these angles by the same amount the training was shifted
    initial_angles = utils.modulo_with_wrapped_range(initial_angles - angle_offsets)
    initial_angles = torch.nan_to_num(initial_angles, nan=0.0)

    # Perform sampling
    sampled_angles_dir = outdir / "sampled_angles"
    sampled_pdb_dir = outdir / "sampled_pdb"
    os.makedirs(sampled_angles_dir, exist_ok=False)
    os.makedirs(sampled_pdb_dir, exist_ok=False)

    sampled_angles = []
    for i in tqdm(range(args.lengths[0], args.lengths[1]), desc="Sampling structures"):
        seed_values = torch.zeros((args.num, 128, 6)).to(device)
        seed_values[:, : args.num_angles, :] = initial_angles
        s = m.sample(
            seed_angles=seed_values,
            seq_lengths=torch.tensor([i for _ in range(args.num)]).to(device),
            num_seed=args.num_angles,
            pbar=False,
        )
        sampled_angles.extend(
            [
                pd.DataFrame(
                    utils.modulo_with_wrapped_range(
                        vals.cpu().numpy() + angle_offsets.cpu().numpy(), -np.pi, np.pi
                    ),
                    columns=ac.EXHAUSTIVE_ANGLES,
                )
                for vals in s
            ]
        )

    # Write the sampled angles and resulting PDB files out
    for i, s in tqdm(enumerate(sampled_angles), desc="Writing sampled angles and PDB structures"):
        s.to_csv(sampled_angles_dir / f"generated_{i}.csv.gz")
        fname = ac.create_new_chain_nerf(str(sampled_pdb_dir / f"generated_{i}.pdb"), s)
        assert fname


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
