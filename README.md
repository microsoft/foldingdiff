# foldingdiff - Diffusion model for protein backbone generation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)

We present a diffusion model for generating novel protein backbone structures. For more details, see our preprint on [arXiv](https://arxiv.org/abs/2209.15611). We also host a trained version of our model on [HuggingFace spaces](https://huggingface.co/spaces/wukevin/foldingdiff) so you can get started with generating protein structures with just your browser!

![Animation of diffusion model protein folds over timesteps](plots/generated_0.gif)

## Installation

To install, clone this using `git clone`. This software is written in Python, notably using PyTorch, PyTorch Lightning, and the HuggingFace transformers library. The required conda environment is defined within the `environment.yml` file. To set this up, make sure you have conda (or [mamba](https://mamba.readthedocs.io/en/latest/index.html)) installed, clone this repository, and run:

```bash
conda env create -f environment.yml
conda activate foldingdiff
pip install -e ./  # make sure ./ is the dir including setup.py
```

### Downloading data

We require some data files not packaged on Git due to their large size. These are not required for sampling (as long as you are not using the `--testcomparison` option, see below); this is required for training your own model. We provide a script in the `data` dir to download requisite CATH data.

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
chmod +x download_cath.sh
./download_cath.sh
```

## Training models

To train your own model on the CATH dataset, use the script at `bin/train.py` in combination with one of the
json config files under `config_jsons` (or write your own). An example usage of this is as follows:

```bash
python bin/train.py config_jsons/cath_full_angles_cosine.json --dryrun
```

By default, the training script will calculate the KL divergence at each timestep before starting training, which can be quite computationally expensive with more timesteps. To skip this, append the `--dryrun` flag. The output of the model will be in the `results` folder with the following major files present:

```
results/
    - config.json           # Contains the config file for the huggingface BERT model itself
    - logs/                 # Contains the logs from training
    - models/               # Contains model checkpoints. By default we store the best 5 models by validation loss and the best 5 by training loss
    - training_args.json    # Full set of arguments, can be used to reproduce run
```

## Pre-trained models

We provide weights for a model trained on the CATH dataset. These weights are stored on HuggingFace model hub at [wukevin/foldingdiff_cath](https://huggingface.co/wukevin/foldingdiff_cath). The following code snippet shows how to load this model, load data (assuming it's been downloaded), and perform a forward pass:

```python
from huggingface_hub import snapshot_download
from torch.utils.data.dataloader import DataLoader
from foldingdiff import modelling
from foldingdiff import datasets as dsets

# Load the model (files will be cached for future calls)
m = modelling.BertForDiffusion.from_dir(snapshot_download("wukevin/foldingdiff_cath"))

# Load dataset
# As part of loading, we try to compute internal angles in parallel. This may
# throw warnings like the following; this is normal.
# WARNING:root:Illegal values for omega in /home/*/projects/foldingdiff-main/data/cath/dompdb/2ebqA00 -- skipping
# After computing these once, the results are saved in a .pkl file under the
# foldingdiff source directory for faster loading in future calls.
clean_dset = dsets.CathCanonicalAnglesOnlyDataset(pad=128, trim_strategy='randomcrop')
noised_dset = dsets.NoisedAnglesDataset(clean_dset, timesteps=1000, beta_schedule='cosine')
dl = DataLoader(noised_dset, batch_size=32, shuffle=False)
x = iter(dl).next()

# Forward pass
predicted_noise = m(x['corrupted'], x['t'], x['attn_mask'])
```

## Sampling protein backbones

To sample protein backbones, use the script `bin/sample.py`. Example commands to do this using the pretrained weights described above are as follows.

```bash
# To sample 10 backbones per length ranging from [50, 128) with a batch size of 512 - reproduces results in our manuscript
python ~/projects/foldingdiff/bin/sample.py -l 50 128 -n 10 -b 512 --device cuda:0
```

This will run the trained model hosted at [wukevin/foldingdiff_cath](https://huggingface.co/wukevin/foldingdiff_cath) and generate sequences of varying lengths. If you wish to load the test dataset and include test chains in the generated plots, use the option `--testcomparison`; note that this requires downloading the CATH dataset, see above. Running `sample.py` will create the following directory structure in the diretory where it is run:

```
some_dir/
    - plots/            # Contains plots comparing the distribution of training/generated angles
    - sampled_angles/   # Contains .csv.gz files with the sampled angles
    - sampled_pdb/      # Contains .pdb files from converting the sampled angles to cartesian coordinates
    - model_snapshot/   # Contains a copy of the model used to produce results
```

Not specifying a `--device` will default to the first device `cuda:0`; use `--device cpu` to run on CPU (though this will be very slow). See the following table for runtimes from our machines.

| Device | Runtime estimates sampling 512 structures |
| --- | --- |
| Nvidia RTX 2080Ti | 7 minutes |
| i9-9960X (16 physical cores) | 2 hours |

### Maximum training similarity TM scores

After generating sequences, we can calculate TM-scores to evaluate the simliarity of the generated sequences and the original sequences. This is done using the script under `bin/tmscore_training.py` and requires data to have been downloaded prior (see above).

### Visualizing diffusion "folding" process

The above sampling code can also be run with the ``--fullhistory`` flag to write an additional subdirectory `sample_history` under each of the `sampled_angles` and `sampled_pdb` folders that contain pdb/csv files coresponding to each timestep in the sampling process. The pdb files, for example, can then be passed into the script under `foldingdiff/pymol_vis.py` to generate a gif of the folding process (as shown above). An example command to do this is:

```bash
python ~/projects/foldingdiff/foldingdiff/pymol_vis.py pdb2gif -i sampled_pdb/sample_history/generated_0/*.pdb -o generated_0.gif
```

**Note** this script lives separately from other plotting code because it depends on PyMOL; feel free to install/activate your own installation of PyMOL for this, or set up an environment using [PyMOL open source](https://github.com/schrodinger/pymol-open-source).

## Evaluating designability of generated backbones

One way to evaluate the quality of generated backbones is via their "designability". This refers to whether or not we can design an amino acid chain that will fold into the designed backbone. To evaluate this, we use an inverse folding model to generate amino acid sequences that are predicted to fold into our generated backbone, and check whether those generated sequences actually fold into a structure comparable to our backbone.

### Inverse folding

Inverse folding is the task of predicting a sequence of amino acids that will produce a given protein backbone structure. We evaluated two different methods for this step, ProteinMPNN and ESM-IF1; we find ProteinMPNN to be significantly more performant. In our analyses, we generate 8 different amino caid sequences for each of FoldingDiff's generated structures.

#### ESM-IF1

We use a different conda environment for [ESM-IF1](https://proceedings.mlr.press/v162/hsu22a.html); see this [Jupyter notebook](https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb) for setup details. We found that the following series of commands works on our machines:

```bash
mamba create -n inverse python=3.9 pytorch cudatoolkit pyg -c pytorch -c conda-forge -c pyg
conda activate inverse
mamba install -c conda-forge biotite
pip install git+https://github.com/facebookresearch/esm.git
```

After this, we `cd` into the folder that contains the `sampled_pdb` directory created by the prior step, and run:

```bash
python ~/projects/foldingdiff/bin/pdb_to_residues_esm.py sampled_pdb -o esm_residues
```

This creates a new folder, `esm_residues` that contains 10 potential residues for each of the pdb files contained in `sampled_pdb`.

#### ProteinMPNN

To set up [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), see the authors guide on their [GitHub](https://github.com/dauparas/ProteinMPNN).

After this, we follow a similar procedure as for ESM-IF1 (above) where we `cd` into the directory containing the `sampled_pdb` folder and run:

```bash
python ~/projects/foldingdiff/bin/pdb_to_residue_proteinmpnn.py sampled_pdb
```

This will create a new directory called `proteinmpnn_residues` containing 8 amino acid chains per sampled PDB structure.

### Structural prediction

After generating amino acid sequences, we check that these recapitulate our original sampled structures by passing them through either OmegaFold or AlphaFold. After running one of these folders, we use the following command to asses self-consistency TM scores:

```bash
python ~/projects/foldingdiff/bin/sctm.py -f alphafold_predictions_proteinmpnn
```

Where `alphafold_predictions_proteinmpnn` is a folder containing the folded structures corresponding to inverse folded amino acid sequences. This produces a json file of all scTM scores, as well as various pdf files containing plots and correlations of the scTM score distribution.

#### OmegaFold

We primarily use [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) to fold the amino acid sequences produced by either ESM-IF1 or ProteinMPNN. This is due to OmegaFold's relatively fast runtime compared to AlphaFold2, and due to the fact that OmegaFold is natively designed to be run without MSA information - making it more suitable for our protein design task.

After creating and activating a separate conda environment and following the authors' instructions for installing OmegaFold, we use the following script to split our input amino acid fasta files across GPUs for inference, and subsequently calculate the self-consistency TM (scTM) scores.

```bash
# Fold each fasta, spreading the work over GPUs 0 and 1, outputs to omegafold_predictions folder
python ~/projects/foldingdiff/bin/omegafold_across_gpus.py esm_residues/*.fasta -g 0 1
```

#### AlphaFold2

We run [AlphaFold2](https://github.com/deepmind/alphafold) via the `localcolabfold` installation method (see [GitHub](https://github.com/YoshitakaMo/localcolabfold)). Due to AlphaFold's runtime requirements, we provide scripts to split the set of fasta files into subdirectories that can then be separately folded; see SLURM script under `scripts/slurm/alphafold.sbatch` for an example.

## Tests

Tests are implemented through a mixture of doctests and unittests. To run unittests, run:

```bash
python -m unittest -v
```

You may see warnings like the following; these are expected.

```bash
WARNING:root:Illegal values for omega in protdiff-main/data/cath/dompdb/5a2qw00 -- skipping
```
