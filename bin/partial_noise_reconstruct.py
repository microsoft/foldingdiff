"""
Partially noise the structures and reconstruct them using the trained model
"""

import os
import argparse
import logging
from pathlib import Path
import json
from typing import *

import numpy as np

import torch
from huggingface_hub import snapshot_download

from foldingdiff import datasets, sampling, utils, modelling


def load_dataset(pdb_files: Collection[str], model_dir: Path):
    """Load dataset"""
    logging.info(f"Loading dataset from {len(pdb_files)} pdb files")
    with open(model_dir / "training_args.json") as source:
        training_args = json.load(source)

    clean_dset_class = {
        "canonical": datasets.CathCanonicalAnglesDataset,
        "canonical-full-angles": datasets.CathCanonicalAnglesOnlyDataset,
        "canonical-minimal-angles": datasets.CathCanonicalMinimalAnglesDataset,
        "cart-coords": datasets.CathCanonicalCoordsDataset,
    }[training_args["angles_definitions"]]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    dset = clean_dset_class(
        pdbs=pdb_files,
        split=None,
        pad=training_args["max_seq_len"],
        min_length=training_args["min_seq_len"],
        trim_strategy="leftalign",
        zero_center=True,  # Offset will be re-specified
        toy=False,
        use_cache=False,
    )
    dset.set_masked_means(np.load(model_dir / "training_mean_offset.npy"))
    noise_dset = datasets.NoisedAnglesDataset(
        dset,
        dset_key="angles",
        timesteps=training_args["timesteps"],
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )
    return noise_dset


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pdb_files", nargs="+", help="PDB files to reconstruct")
    parser.add_argument("output_json", type=str, help="Output JSON file")
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=800,
        help="Timesteps (max 1000) of noise to add",
    )
    parser.add_argument(
        "-m", "--model", default="wukevin/foldingdiff_cath", help="Model to use"
    )
    parser.add_argument("-d", "--device", type=int, default=1, help="GPU to use")
    return parser


def get_reconstruction_error(
    pdb_files: Collection[str],
    timesteps: int = 800,
    model: str = "wukevin/foldingdiff_cath",
    device: torch.device = torch.device("cuda:1"),
) -> np.ndarray:
    """Get the reconstruction error for a set of PDB files"""
    if utils.is_huggingface_hub_id(model):
        logging.info(f"Detected huggingface repo ID {model}")
        dl_path = snapshot_download(model)  # Caching is automatic
        assert os.path.isdir(dl_path)
        logging.info(f"Using downloaded model at {dl_path}")
        model = dl_path

    assert os.path.isdir(model), f"Model path {model} does not exist"
    model = Path(model)

    # Load in the dataset
    dset = load_dataset(pdb_files, model)

    # Load in model
    model = modelling.BertForDiffusionBase.from_dir(model).to(torch.device(device))

    # Run the partial denoising
    scores_wrt_coords, _scores_wrt_angles = sampling.get_reconstruction_error(
        model,
        dset=dset,
        noise_timesteps=timesteps,
    )
    logging.info(
        f"Reconstuction scores from t={timesteps}: {(np.min(scores_wrt_coords), np.max(scores_wrt_coords))}"
    )
    return scores_wrt_coords


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    scores = get_reconstruction_error(
        args.pdb_files,
        timesteps=args.timesteps,
        model=args.model,
        device=torch.device(f"cuda:{args.device}"),
    )

    scores_dict = {pdb: score for pdb, score in zip(args.pdb_files, scores)}
    with open(args.output_json, "w") as sink:
        json.dump(
            {"timesteps": args.timesteps, "model": args.model, "tmscores": scores_dict}, sink, indent=4
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
