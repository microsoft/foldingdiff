# Protein diffusion

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We present a diffusion model for generating novel protein backbone structures.

![Animation of diffusion model protein folds over timesteps](plots/generated_0.gif)

## Installation

To install, clone this using `git clone` followed by `git lfs fetch`. Note that this requires [git-lfs](https://git-lfs.github.com) to be installed on your system.

This software is written in Python, notably using PyTorch, PyTorch Lightning, and the HuggingFace
transformers library. The required conda environment is defined within the `environment.yml` file. To set this up, make sure you have conda (or [mamba](https://mamba.readthedocs.io/en/latest/index.html)) installed, clone this repository, and run:

```bash
conda env create -f environment.yml
```

## Training models

To train a model on the CATH dataset, use the script at `bin/train.py` in combination with one of the
json config files under `config_jsons` (or write your own). An example usage of this is as follows:

```bash
python bin/train.py config_jsons/full_run_canonical_angles_only_zero_centered_1000_timesteps_reduced_len.json
```

The output of the model will be in the `results` folder with the following major files present:

```
results/
    - config.json           # Contains the config file for the huggingface BERT model itself
    - logs/                 # Contains the logs from training
    - models/               # Contains model checkpoints. By default we store the best 5 models by validation loss and the best 5 by training loss
    - training_args.json    # Full set of arguments, can be used to reproduce run
```

## Pre-trained models

We provide weihts for a model trained on the CATH dataset. These weights are located under the `models/cath_pretrained` directory and are stored via Git LFS. To programmatically load these weights, you can use code defined under `protdiff/modelling.py` as such:

```python
import modelling

modelling.BertForDiffusion.from_dir("models/cath_pretrained").to(torch.device("cuda:0"))
```

Providing this path to premade script such as for sampling is detailed below.

## Downloading data

We requires some data files not packaged on Git due to their large size. These are required to be downloaded locally even if you are not training and are only sampling. The simple command to do this is as follows:

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
./download_cath.sh
```

## Sampling protein backbones

To sample protein backbones, use the script `bin/sample.py`. Example commands to do this using the pretrained weights described above is as follows.

```bash
# To sample 256 backbones
python ~/projects/protdiff/bin/sample.py ~/projects/protdiff/models/cath_pretrained --num 256 --device cuda:3
# To sample 10 backbones per length ranging from [50, 128) - this reproduces results in our manuscript
python ~/projects/protdiff/bin/sample.py ~/projects/protdiff/models/cath_pretrained -l sweep --device cuda:3
```

This will run the model contained in the `results` folder and generate 512 sequences of varying lengths. Not specifying a device will default to the first device `cuda:0`; use `--device cpu` to run on CPU. This will create the following directory structure in the diretory where it is run:

```
some_dir/
    - plots/            # Contains plots comparing the distribution of training/generated angles
    - sampled_angles/   # Contains .csv.gz files with the sampled angles
    - sampled_pdb/      # Contains .pdb files from converting the sampled angles to cartesian coordinates
    - model_snapshot/   # Contains a copy of the model used to produce results
```

### Maximum training similarity TM scores

After generating sequences, we can calculate TM-scores to evaluate the simliarity of the generated sequences and the original sequences. This is done using the script under `bin/tmscore_training.py`.

### Visualizing diffusion "folding" process

The above sampling code can also be run with the ``--fullhistory`` flag to write an additional subdirectory `sample_history` under each of the `sampled_angles` and `sampled_pdb` folders that contain pdb/csv files coresponding to each timestep in the sampling process. The pdb files, for example, can then be passed into the script under `protdiff/pymol_vis.py` to generate a gif of the folding process (as shown above). An example command to do this is:

```bash
python ~/protdiff/protdiff/pymol_vis.py pdb2gif -i sampled_pdb/sample_history/generated_0/*.pdb -o generated_0.gif
```

**Note** this script lives separately from other plotting code because it depends on PyMOL; feel free to install/activate your own installation of PyMOL for this.

## Evaluating designability of generated backbones

One way to evaluate the quality of generated backbones is via their "designability". This refers to whether or not we can design an amino acid chain that will fold into the designed backbone. To evaluate this, we use the ESM inverse folding model to generate residues that are predicted to fold into our generated backbone, and use OmegaFold to check whether that generated sequence actually does fold into a structure comparable to our backbone. (While prior backbone design works have used AlphaFold2 for their designability evaluations, this was previously done without providing AlphaFold with MSA information; OmegaFold is designed from the ground up to use sequence only, and is therefore better suited for this use case.)

### Inverse folding with ESM

We use a different conda environment for this step; see <https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb> for setup details. We found that the following command works on our machines:

```bash
mamba create -n inverse python=3.9 pytorch cudatoolkit pyg -c pytorch -c conda-forge -c pyg
conda activate inverse
mamba install -c conda-forge biotite
pip install git+https://github.com/facebookresearch/esm.git
```

After this, we `cd` into the folder that contains the `sampled_pdb` directory created by the prior step, and run:

```bash
python ~/protdiff/bin/pdb_to_residues_esm.py sampled_pdb -o esm_residues
```

This creates a new folder, `esm_residues` that contains 10 potential residues for each of the pdb files contained in `sampled_pdb`.

### Structural prediction with OmegaFold

We use [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) to fold the amino acid sequences produced above. After creating and activating a separate conda environment and following the authors' instructions for installing OmegaFold, we use the following script to split our input amino acid fasta files across GPUs for inference, and subsequently calculate the self-consistency TM (scTM) scores.

```bash
# Fold each fasta, spreading the work over GPUs 0 and 1, outputs to omegafold_predictions folder
python ~/projects/protdiff/bin/omegafold_across_gpus.py esm_residues/*.fasta -g 0 1
# Calculate the scTM scores; parallelizes across all CPUs
python ~/projects/protdiff/bin/omegafold_self_tm.py  # Requires no arguments
```

After executing these commands, the final command produces a json file of all scmtm scores, as well as various pdf files containing plots and correlations of the scTM score distribution.

## Tests

Tests are implemented through a mixture of doctests and unittests. To run unittests, run:

```bash
python -m unittest -v
```
