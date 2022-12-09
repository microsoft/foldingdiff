"""
Training script.

Example usage: python ~/protdiff/bin/train.py ~/protdiff/config_jsons/full_run_canonical_angles_only_zero_centered_1000_timesteps_reduced_len.json
"""

import os, sys
import shutil
import json
import logging
from pathlib import Path
import multiprocessing
import argparse
import functools
from datetime import datetime
from typing import *

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from transformers import BertConfig

from foldingdiff import datasets
from foldingdiff import modelling
from foldingdiff import losses
from foldingdiff import beta_schedules
from foldingdiff import plotting
from foldingdiff import utils
from foldingdiff import custom_metrics as cm

assert torch.cuda.is_available(), "Requires CUDA to train"
# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Define some typing literals
ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords"
]


@pl.utilities.rank_zero_only
def plot_timestep_distributions(
    train_dset,
    timesteps: int,
    plots_folder: Path,
    shift_angles_zero_twopi: bool = False,
    n_intervals: int = 11,
) -> None:
    """
    Plot the distributions across timesteps. This is parallelized across multiple cores
    """
    ts = np.linspace(0, timesteps, num=n_intervals, endpoint=True).astype(int)
    ts = np.minimum(ts, timesteps - 1).tolist()
    logging.info(f"Plotting distributions at {ts} to {plots_folder}")
    args = [
        (
            t,
            train_dset,
            True,
            not shift_angles_zero_twopi,
            plots_folder / f"train_dists_at_t_{t}.pdf",
        )
        for t in ts
    ]

    # Parallelize the plotting
    pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(ts)))
    pool.starmap(plotting.plot_val_dists_at_t, args)
    pool.close()
    pool.join()


@pl.utilities.rank_zero_only
def plot_kl_divergence(train_dset, plots_folder: Path) -> None:
    """
    Plot the KL divergence over time
    """
    # This works because the main body of this script should clean out the dir
    # between runs
    outname = plots_folder / "kl_divergence_timesteps.pdf"
    if outname.is_file():
        logging.info(f"KL divergence plot exists at {outname}; skipping...")
    kl_at_timesteps = cm.kl_from_dset(train_dset)  # Shape (n_timesteps, n_features)
    n_timesteps, n_features = kl_at_timesteps.shape
    fig, axes = plt.subplots(
        dpi=300, figsize=(n_features * 3.05, 2.5), ncols=n_features, sharey=True
    )
    for i, (ft_name, ax) in enumerate(zip(train_dset.feature_names["angles"], axes)):
        ax.plot(np.arange(n_timesteps), kl_at_timesteps[:, i], label=ft_name)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set(title=ft_name)
        if i == 0:
            ax.set(ylabel="KL divergence")
        ax.set(xlabel="Timestep")
    fig.suptitle(
        f"KL(empirical || Gaussian) over timesteps={train_dset.timesteps}", y=1.05
    )
    fig.savefig(outname, bbox_inches="tight")


def get_train_valid_test_sets(
    dataset_key: str = "cath",
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 512,
    min_seq_len: int = 0,
    seq_trim_strategy: datasets.TRIM_STRATEGIES = "leftalign",
    timesteps: int = 250,
    variance_schedule: beta_schedules.SCHEDULES = "linear",
    var_scale: float = np.pi,
    toy: Union[int, bool] = False,
    exhaustive_t: bool = False,
    syn_noiser: str = "",
    single_angle_debug: int = -1,  # Noise and return a single angle. -1 to disable, 1-3 for omega/theta/phi
    single_time_debug: bool = False,  # Noise and return a single time
    train_only: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the dataset objects to use for train/valid/test

    Note, these need to be wrapped in data loaders later
    """
    assert (
        single_angle_debug != 0
    ), f"Invalid value for single_angle_debug: {single_angle_debug}"

    clean_dset_class = {
        "canonical": datasets.CathCanonicalAnglesDataset,
        "canonical-full-angles": datasets.CathCanonicalAnglesOnlyDataset,
        "canonical-minimal-angles": datasets.CathCanonicalMinimalAnglesDataset,
        "cart-coords": datasets.CathCanonicalCoordsDataset,
    }[angles_definitions]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    splits = ["train"] if train_only else ["train", "validation", "test"]
    logging.info(f"Creating data splits: {splits}")
    clean_dsets = [
        clean_dset_class(
            pdbs=dataset_key,
            split=s,
            pad=max_seq_len,
            min_length=min_seq_len,
            trim_strategy=seq_trim_strategy,
            zero_center=False if angles_definitions == "cart-coords" else True,
            toy=toy,
        )
        for s in splits
    ]
    assert len(clean_dsets) == len(splits)
    # Set the training set mean to the validation set mean
    if len(clean_dsets) > 1 and clean_dsets[0].means is not None:
        logging.info(f"Updating valid/test mean offset to {clean_dsets[0].means}")
        for i in range(1, len(clean_dsets)):
            clean_dsets[i].means = clean_dsets[0].means

    if syn_noiser != "":
        if syn_noiser == "halfhalf":
            logging.warning("Using synthetic half-half noiser")
            dset_noiser_class = datasets.SynNoisedByPositionDataset
        else:
            raise ValueError(f"Unknown synthetic noiser {syn_noiser}")
    else:
        if single_angle_debug > 0:
            logging.warning("Using single angle noise!")
            dset_noiser_class = functools.partial(
                datasets.SingleNoisedAngleDataset, ft_idx=single_angle_debug
            )
        elif single_time_debug:
            logging.warning("Using single angle and single time noise!")
            dset_noiser_class = datasets.SingleNoisedAngleAndTimeDataset
        else:
            dset_noiser_class = datasets.NoisedAnglesDataset

    logging.info(f"Using {dset_noiser_class} for noise")
    noised_dsets = [
        dset_noiser_class(
            dset=ds,
            dset_key="coords" if angles_definitions == "cart-coords" else "angles",
            timesteps=timesteps,
            exhaustive_t=(i != 0) and exhaustive_t,
            beta_schedule=variance_schedule,
            nonangular_variance=1.0,
            angular_variance=var_scale,
        )
        for i, ds in enumerate(clean_dsets)
    ]
    for dsname, ds in zip(splits, noised_dsets):
        logging.info(f"{dsname}: {ds}")

    # Pad with None values
    if len(noised_dsets) < 3:
        noised_dsets = noised_dsets + [None] * int(3 - len(noised_dsets))
    assert len(noised_dsets) == 3

    return tuple(noised_dsets)


def build_callbacks(
    outdir: str, early_stop_patience: Optional[int] = None, swa: bool = False
):
    """
    Build out the callbacks
    """
    # Create the logging dir
    os.makedirs(os.path.join(outdir, "logs/lightning_logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models/best_by_valid"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models/best_by_train"), exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(outdir, "models/best_by_valid"),
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=os.path.join(outdir, "models/best_by_train"),
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
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


# For some arg defaults, see as reference:
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer.html


@pl.utilities.rank_zero_only
def record_args_and_metadata(func_args: Dict[str, Any], results_folder: Path):
    # Create results directory
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(exist_ok=True)
    with open(results_folder / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args, sink, indent=4)
        for k, v in func_args.items():
            logging.info(f"Training argument: {k}={v}")

    # Record current Git version
    try:
        import git

        repo = git.Repo(
            path=os.path.dirname(os.path.abspath(__file__)),
            search_parent_directories=True,
        )
        sha = repo.head.object.hexsha
        with open(results_folder / "git_sha.txt", "w") as sink:
            sink.write(sha + "\n")
    except git.exc.InvalidGitRepositoryError:
        logging.warning("Could not determine Git repo status -- not a git repo")
    except ModuleNotFoundError:
        logging.warning(
            f"Could not determine Git repo status -- GitPython is not installed"
        )


def train(
    # Controls output
    results_dir: str = "./results",
    # Controls data loading and noising process
    dataset_key: str = "cath",  # cath, alhpafold, or a directory containing pdb files
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 512,
    min_seq_len: int = 0,  # 0 means no filtering based on min sequence length
    trim_strategy: datasets.TRIM_STRATEGIES = "leftalign",
    timesteps: int = 250,
    variance_schedule: beta_schedules.SCHEDULES = "linear",  # cosine better on single angle toy test
    variance_scale: float = 1.0,
    # Related to model architecture
    time_encoding: modelling.TIME_ENCODING = "gaussian_fourier",
    num_hidden_layers: int = 12,  # Default 12
    hidden_size: int = 384,  # Default 768
    intermediate_size: int = 768,  # Default 3072
    num_heads: int = 12,  # Default 12
    position_embedding_type: Literal[
        "absolute", "relative_key", "relative_key_query"
    ] = "absolute",  # relative_key = https://arxiv.org/pdf/1803.02155.pdf | relative_key_query = https://arxiv.org/pdf/2009.13658.pdf
    dropout_p: float = 0.1,  # Default 0.1, can disable for debugging
    decoder: modelling.DECODER_HEAD = "mlp",
    # Related to training strategy
    gradient_clip: float = 1.0,  # From BERT trainer
    batch_size: int = 64,
    lr: float = 5e-5,  # Default lr for huggingface BERT trainer
    loss: modelling.LOSS_KEYS = "smooth_l1",
    use_pdist_loss: Union[
        float, Tuple[float, float]
    ] = 0.0,  # Use the pairwise distances between CAs as an additional loss term, multiplied by this scalar
    l2_norm: float = 0.0,  # AdamW default has 0.01 L2 regularization, but BERT trainer uses 0.0
    l1_norm: float = 0.0,
    circle_reg: float = 0.0,
    min_epochs: Optional[int] = None,
    max_epochs: int = 10000,
    early_stop_patience: int = 0,  # Set to 0 to disable early stopping
    lr_scheduler: modelling.LR_SCHEDULE = None,
    use_swa: bool = False,  # Stochastic weight averaging can improve training genearlization
    # Misc. and debugging
    multithread: bool = True,
    subset: Union[bool, int] = False,  # Subset to n training examples
    exhaustive_validation_t: bool = False,  # Exhaustively enumerate t for validation/test
    syn_noiser: str = "",  # If specified, use a synthetic noiser
    single_angle_debug: int = -1,  # Noise and return a single angle, choose [1, 2, 3] or -1 to disable
    single_timestep_debug: bool = False,  # Noise and return a single timestep
    cpu_only: bool = False,
    ngpu: int = -1,  # -1 for all GPUs
    write_valid_preds: bool = False,  # Write validation predictions to disk at each epoch
    dryrun: bool = False,  # Disable some frills for a fast run to just train
):
    """Main training loop"""
    # Record the args given to the function before we create more vars
    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
    func_args = locals()

    results_folder = Path(results_dir)
    record_args_and_metadata(func_args, results_folder)

    # Get datasets and wrap them in dataloaders
    dsets = get_train_valid_test_sets(
        dataset_key=dataset_key,
        angles_definitions=angles_definitions,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        seq_trim_strategy=trim_strategy,
        timesteps=timesteps,
        variance_schedule=variance_schedule,
        var_scale=variance_scale,
        toy=subset,
        syn_noiser=syn_noiser,
        exhaustive_t=exhaustive_validation_t,
        single_angle_debug=single_angle_debug,
        single_time_debug=single_timestep_debug,
    )
    # Record the masked means in the output directory
    np.save(
        results_folder / "training_mean_offset.npy",
        dsets[0].dset.get_masked_means(),
        fix_imports=False,
    )

    # Calculate effective batch size
    # https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html#batch-size
    # Under DDP, effective batch size is batch_size * num_gpus * num_nodes
    effective_batch_size = batch_size
    if torch.cuda.is_available():
        effective_batch_size = int(batch_size / torch.cuda.device_count())
    pl.utilities.rank_zero_info(
        f"Given batch size: {batch_size} --> effective batch size with {torch.cuda.device_count()} GPUs: {effective_batch_size}"
    )

    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=i == 0,  # Shuffle only train loader
            num_workers=multiprocessing.cpu_count() if multithread else 1,
            pin_memory=True,
        )
        for i, ds in enumerate(dsets)
    ]

    # Create plots in output directories of distributions from different timesteps
    plots_folder = results_folder / "plots"
    os.makedirs(plots_folder, exist_ok=True)
    # Skip this for debug runs
    if (
        single_angle_debug < 0
        and not single_timestep_debug
        and not syn_noiser
        and not dryrun
    ):
        plot_kl_divergence(dsets[0], plots_folder)
        plot_timestep_distributions(
            dsets[0],
            timesteps=timesteps,
            plots_folder=plots_folder,
        )

    # https://jaketae.github.io/study/relative-positional-encoding/
    # looking at the relative distance between things is more robust

    loss_fn = loss
    if single_angle_debug > 0 or single_timestep_debug or syn_noiser:
        loss_fn = functools.partial(losses.radian_smooth_l1_loss, beta=0.1 * np.pi)
    logging.info(f"Using loss function: {loss_fn}")

    # Shape of the input is (batch_size, timesteps, features)
    sample_input = dsets[0][0]["corrupted"]  # First item of the training dset
    model_n_inputs = sample_input.shape[-1]
    logging.info(f"Auto detected {model_n_inputs} inputs")

    cfg = BertConfig(
        max_position_embeddings=max_seq_len,
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        position_embedding_type=position_embedding_type,
        hidden_dropout_prob=dropout_p,
        attention_probs_dropout_prob=dropout_p,
        use_cache=False,
    )
    # ft_is_angular from the clean datasets angularity definition
    ft_key = "coords" if angles_definitions == "cart-coords" else "angles"
    model = modelling.BertForDiffusion(
        config=cfg,
        time_encoding=time_encoding,
        decoder=decoder,
        ft_is_angular=dsets[0].dset.feature_is_angular[ft_key],
        ft_names=dsets[0].dset.feature_names[ft_key],
        lr=lr,
        loss=loss_fn,
        use_pairwise_dist_loss=use_pdist_loss
        if isinstance(use_pdist_loss, float)
        else [*use_pdist_loss, timesteps],
        l2=l2_norm,
        l1=l1_norm,
        circle_reg=circle_reg,
        epochs=max_epochs,
        steps_per_epoch=len(train_dataloader),
        lr_scheduler=lr_scheduler,
        write_preds_to_dir=results_folder / "valid_preds"
        if write_valid_preds
        else None,
    )
    cfg.save_pretrained(results_folder)

    callbacks = build_callbacks(
        outdir=results_folder, early_stop_patience=early_stop_patience, swa=use_swa
    )

    # Get accelerator and distributed strategy
    accelerator, strategy = "cpu", None
    if not cpu_only and torch.cuda.is_available():
        accelerator = "cuda"
        if torch.cuda.device_count() > 1:
            # https://github.com/Lightning-AI/lightning/discussions/6761https://github.com/Lightning-AI/lightning/discussions/6761
            strategy = DDPStrategy(find_unused_parameters=False)

    logging.info(f"Using {accelerator} with strategy {strategy}")
    trainer = pl.Trainer(
        default_root_dir=results_folder,
        gradient_clip_val=gradient_clip,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=results_folder / "logs"),
        log_every_n_steps=min(200, len(train_dataloader)),  # Log >= once per epoch
        accelerator=accelerator,
        strategy=strategy,
        gpus=ngpu,
        enable_progress_bar=False,
        move_metrics_to_cpu=False,  # Saves memory
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
    plotting.plot_losses(
        metrics_csv, out_fname=plots_folder / "losses.pdf", simple=True
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=None,
        help="Use a toy dataset of n items rather than full dataset",
    )
    parser.add_argument(
        "--debug_single_time",
        action="store_true",
        help="Debug single angle and timestep",
    )
    parser.add_argument("--cpu", action="store_true", help="Force use CPU")
    parser.add_argument(
        "--ngpu", type=int, default=-1, help="Number of GPUs to use (-1 for all)"
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run")
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
    config_args = utils.update_dict_nonnull(
        config_args,
        {
            "results_dir": args.outdir,
            "subset": args.toy,
            "single_timestep_debug": args.debug_single_time,
            "cpu_only": args.cpu,
            "ngpu": args.ngpu,
            "dryrun": args.dryrun,
        },
    )
    train(**config_args)


if __name__ == "__main__":
    curr_time = datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"training_{curr_time}.log"),
            logging.StreamHandler(),
        ],
    )

    main()
