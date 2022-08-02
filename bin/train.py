"""
Training script
"""

import os, sys
import shutil
import json
import logging
from pathlib import Path
import multiprocessing
import argparse
from typing import *

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertConfig

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))

import datasets
import modelling
import losses
from beta_schedules import SCHEDULES
import plotting
import utils


# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


def plot_epoch_losses(loss_values, fname: str):
    """Plot the loss values and save to fname"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(np.arange(len(loss_values)), loss_values)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")
    fig.savefig(fname)


def get_train_valid_test_sets(
    timesteps: int,
    variance_schedule: SCHEDULES,
    noise_prior: Literal["gaussian", "uniform"] = "gaussian",
    adaptive_noise_mean_var: bool = True,
    shift_to_zero_twopi: bool = True,
    toy: Union[int, bool] = False,
    exhaustive_t: bool = False,
    single_angle_debug: bool = False,  # Noise and retur a single angle
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the dataset objects to use for train/valid/test

    Note, these need to be wrapped in data loaders later
    """
    clean_dsets = [
        datasets.CathConsecutiveAnglesDataset(
            split=s, shift_to_zero_twopi=shift_to_zero_twopi, toy=toy
        )
        for s in ["train", "validation", "test"]
    ]

    if noise_prior == "gaussian":
        if single_angle_debug:
            logging.warning("Using single angle noise!")
            dset_noiser_class = datasets.SingleNoisedAngleDataset
        else:
            dset_noiser_class = datasets.NoisedAnglesDataset
    elif noise_prior == "uniform":
        dset_noiser_class = datasets.GaussianDistUniformAnglesNoisedDataset
    else:
        raise ValueError(f"Unrecognized noise prior: {noise_prior}")

    logging.info(f"Using {dset_noiser_class.__name__} for noise")
    noised_dsets = [
        dset_noiser_class(
            ds,
            dset_key="angles",
            timesteps=timesteps,
            exhaustive_t=(i != 0) and exhaustive_t,
            beta_schedule=variance_schedule,
            modulo=(
                [0, 2 * np.pi, 2 * np.pi, 2 * np.pi] if shift_to_zero_twopi else None
            ),
            noise_by_modulo=adaptive_noise_mean_var,
        )
        for i, ds in enumerate(clean_dsets)
    ]
    return tuple(noised_dsets)


def build_callbacks(early_stop_patience: Optional[int] = None, swa: bool = False):
    """
    Build out the callbacks
    """
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss", save_top_k=1, save_weights_only=True,
        ),
    ]
    if early_stop_patience is not None and early_stop_patience > 0:
        logging.info(f"Using early stopping with patience {early_stop_patience}")
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                verbose=True,
                mode="min",
            )
        )
    if swa:
        # Stochastic weight averaging
        callbacks.append(pl.callbacks.StochasticWeightAveraging())
    logging.info(f"Model callbacks: {callbacks}")
    return callbacks


def train(
    # Controls output
    results_dir: str = "./results",
    # Controls data loading and noising process
    shift_angles_zero_twopi: bool = True,
    noise_prior: Literal["gaussian", "uniform"] = "gaussian",  # Uniform not tested
    timesteps: int = 1000,
    variance_schedule: SCHEDULES = "linear",
    adaptive_noise_mean_var: bool = True,
    # Related to model architecture
    time_encoding: Literal["gaussian_fourier", "sinusoidal"] = "sinusoidal",
    num_hidden_layers: int = 6,  # Default 12
    hidden_size: int = 72,  # Default 768
    intermediate_size: int = 144,  # Default 3072
    num_heads: int = 8,  # Default 12
    position_embedding_type: Literal[
        "absolute", "relative_key", "relative_key_query"
    ] = "relative_key_query",
    # Related to training strategy
    gradient_clip: float = 0.5,
    batch_size: int = 64,
    lr: float = 1e-3,
    loss: Literal["huber", "radian_l1", "radian_l1_smooth"] = "radian_l1_smooth",
    l2_norm: float = 0.01,  # AdamW default has 0.01 L2 regularization
    l1_norm: float = 0.0,
    min_epochs: int = 500,
    max_epochs: int = 2000,
    early_stop_patience: int = 10,  # Set to 0 to disable early stopping
    use_swa: bool = False,  # Stochastic weight averaging can improve training genearlization
    # Misc.
    multithread: bool = True,
    subset: Union[bool, int] = False,  # Subset to n training examples
    exhaustive_validation_t: bool = False,  # Exhaustively enumerate t for validation/test
    single_angle_debug: bool = False,  # Noise and return a single angle
):
    """Main training loop"""
    # Record the args given to the function before we create more vars
    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
    func_args = locals()

    # Create results directory
    results_folder = Path(results_dir)
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(exist_ok=True)
    with open(results_folder / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args, sink, indent=4)
        for k, v in func_args.items():
            logging.info(f"Training argument: {k}={v}")

    # Get datasets and wrap them in dataloaders
    dsets = get_train_valid_test_sets(
        timesteps=timesteps,
        variance_schedule=variance_schedule,
        noise_prior=noise_prior,
        adaptive_noise_mean_var=adaptive_noise_mean_var,
        shift_to_zero_twopi=shift_angles_zero_twopi,
        toy=subset,
        exhaustive_t=exhaustive_validation_t,
        single_angle_debug=single_angle_debug,
    )
    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=i == 0,  # Shuffle only train loader
            num_workers=multiprocessing.cpu_count() if multithread else 1,
        )
        for i, ds in enumerate(dsets)
    ]

    # Create plots in output directories of distributions from different timesteps
    plots_folder = results_folder / "plots"
    os.makedirs(plots_folder, exist_ok=True)
    if not single_angle_debug:  # Skip this for debug runs
        for t in np.linspace(0, timesteps, num=11, endpoint=True).astype(int):
            t = min(t, timesteps - 1)  # Ensure we don't exceed the number of timesteps
            logging.info(f"Plotting distribution at time {t}")
            plotting.plot_val_dists_at_t(
                dsets[0],
                t=t,
                share_axes=False,
                zero_center_angles=not shift_angles_zero_twopi,
                fname=plots_folder / f"train_dists_at_t_{t}.pdf",
            )

    # https://jaketae.github.io/study/relative-positional-encoding/
    # looking at the relative distance between things is more robust
    cfg = BertConfig(
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        position_embedding_type=position_embedding_type,
    )
    model = modelling.BertForDiffusion(
        cfg,
        time_encoding=time_encoding,
        n_inputs=1 if single_angle_debug else 4,
        lr=lr,
        loss=loss if not single_angle_debug else losses.radian_smooth_l1_loss,
        l2=l2_norm,
        l1=l1_norm,
    )
    cfg.save_pretrained(results_folder)

    callbacks = build_callbacks(early_stop_patience=early_stop_patience, swa=use_swa)
    trainer = pl.Trainer(
        default_root_dir=results_folder,
        gradient_clip_val=gradient_clip,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=results_folder / "logs"),
        log_every_n_steps=min(50, len(train_dataloader)),  # Log at least once per epoch
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Plot the losses
    metrics_csv = os.path.join(
        trainer.logger.save_dir, "lightning_logs/version_0/metrics.csv"
    )
    assert os.path.isfile(metrics_csv)
    # Plot the losses
    plotting.plot_losses(metrics_csv, out_fname=plots_folder / "losses.pdf")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
    parser.add_argument(
        "config", nargs="?", default="", type=str, help="json of params"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to write model training outputs",
    )
    parser.add_argument(
        "--toy",
        type=int,
        default=0,
        help="Use a toy dataset of n items rather than full dataset",
    )
    parser.add_argument(
        "--debug_single", action="store_true", help="Debug single angle"
    )
    return parser


def main():
    """Run the training script based on params in the given json file"""
    parser = build_parser()
    args = parser.parse_args()

    # Load in parameters and run training loop
    config_args = {}  # Empty dictionary as default
    if args.config:
        with open(args.config) as source:
            config_args = json.load(source)
    config_args = utils.update_dict(
        config_args,
        {
            "results_dir": args.outdir,
            "subset": args.toy,
            "single_angle_debug": args.debug_single,
        },
    )
    train(**config_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
