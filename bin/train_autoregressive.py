"""
Train a baseline autoregressive model that uses a causal LM approach to generating
series of angles
"""
import os
from pathlib import Path
import json
import argparse
from datetime import datetime
import logging
import multiprocessing
from typing import *

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from transformers import BertConfig

from foldingdiff import datasets, modelling, losses, plotting, utils
from foldingdiff import custom_metrics as cm

from train import ANGLES_DEFINITIONS, build_callbacks, record_args_and_metadata


def get_train_valid_test_sets(
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 512,
    min_seq_len: int = 0,
    seq_trim_strategy: datasets.TRIM_STRATEGIES = "leftalign",
) -> Tuple[
    datasets.AutoregressiveCausalDataset,
    datasets.AutoregressiveCausalDataset,
    datasets.AutoregressiveCausalDataset,
]:
    """
    Get the train/valid/test splits using the autoregressive wrapper on the datsets
    """

    clean_dset_class = {
        "canonical": datasets.CathCanonicalAnglesDataset,
        "canonical-full-angles": datasets.CathCanonicalAnglesOnlyDataset,
        "canonical-minimal-angles": datasets.CathCanonicalMinimalAnglesDataset,
        "cart-coords": datasets.CathCanonicalCoordsDataset,
    }[angles_definitions]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    splits = ["train", "validation", "test"]
    logging.info(f"Creating data splits: {splits}")
    clean_dsets = [
        clean_dset_class(
            split=s,
            pad=max_seq_len,
            min_length=min_seq_len,
            trim_strategy=seq_trim_strategy,
            zero_center=False if angles_definitions == "cart-coords" else True,
        )
        for s in splits
    ]

    # Set the training set mean to the validation set mean
    if len(clean_dsets) > 1 and clean_dsets[0].means is not None:
        logging.info(f"Updating valid/test mean offset to {clean_dsets[0].means}")
        for i in range(1, len(clean_dsets)):
            clean_dsets[i].means = clean_dsets[0].means

    causal_dsets = [
        datasets.AutoregressiveCausalDataset(
            d, dset_key="coords" if angles_definitions == "cart-coords" else "angles"
        )
        for d in clean_dsets
    ]
    for dsname, ds in zip(splits, causal_dsets):
        logging.info(f"{dsname}: {ds}")
    return causal_dsets


def train(
    results_dir: str = "./results",
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 128,
    min_seq_len: int = 40,
    trim_strategy: datasets.TRIM_STRATEGIES = "randomcrop",
    # Related to model architecture
    seq_len_encoding: modelling.TIME_ENCODING = "gaussian_fourier",  # Embeds the total sequence length
    num_hidden_layers: int = 12,  # Default 12
    hidden_size: int = 384,  # Default 768
    intermediate_size: int = 768,  # Default 3072
    num_heads: int = 12,  # Default 12
    position_embedding_type: Literal[
        "absolute", "relative_key_query", "relative_key"
    ] = "absolute",  # Default absolute
    dropout_p: float = 0.1,
    decoder: modelling.DECODER_HEAD = "mlp",
    # Related to training strategy
    gradient_clip: float = 1.0,
    batch_size: int = 256,
    lr: float = 5e-5,
    l2_norm: float = 0.01,
    loss: modelling.LOSS_KEYS = "smooth_l1",
    min_epochs: Optional[int] = None,
    max_epochs: int = 10000,  # 10000, set to 100 for debug
    early_stop_patience: int = 0,  # Set to 0 to disable early stopping
    lr_scheduler: modelling.LR_SCHEDULE = "LinearWarmup",  # Try LinearWarmup?
    use_swa: bool = False,
):
    """
    Train the model
    """
    func_args = locals()

    results_folder = Path(results_dir)
    record_args_and_metadata(func_args, results_folder)

    plots_folder = results_folder / "plots"
    os.makedirs(plots_folder, exist_ok=True)

    ft_key = "coords" if angles_definitions == "cart-coords" else "angles"
    dsets = get_train_valid_test_sets(
        angles_definitions=angles_definitions,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        seq_trim_strategy=trim_strategy,
    )
    assert len(dsets) == 3
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

    # Create data loaders
    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=i == 0,  # Shuffle only train loader
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )
        for i, ds in enumerate(dsets)
    ]

    logging.info(f"Using loss function: {loss}")

    # Shape of the input is (batch_size, timesteps, features)
    sample_input = dsets[0][0][ft_key]  # First item of the training dset
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
    cfg.save_pretrained(results_folder)

    model = modelling.BertForAutoregressive(
        config=cfg,
        time_encoding=seq_len_encoding,  # Repurpose the time embedder to do sequence instead
        decoder=decoder,
        ft_is_angular=dsets[0].dset.feature_is_angular[ft_key],
        ft_names=dsets[0].dset.feature_names[ft_key],
        lr=lr,
        loss_key=loss,
        l2=l2_norm,
        epochs=max_epochs,
        steps_per_epoch=len(train_dataloader),
        lr_scheduler=lr_scheduler,
    )

    callbacks = build_callbacks(
        outdir=results_folder, early_stop_patience=early_stop_patience, swa=use_swa
    )

    # Get accelerator and distributed strategy
    accelerator, strategy = "cpu", None
    if torch.cuda.is_available():
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
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
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
    return parser


def main():
    """
    Run the training script based on params in the given json file
    """
    parser = build_parser()
    args = parser.parse_args()

    # Load in parameters and run training loop
    config_args = {}
    if args.config:
        with open(args.config, "r") as f:
            config_args = json.load(f)

    config_args = utils.update_dict_nonnull(config_args, {"results_dir": args.outdir})

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
