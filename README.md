# Protein diffusion

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We present a diffusion model for generating novel protein backbone structures.

## Installation
The required conda environment is defined within the `environment.yml` file. To set this up, make
sure you have conda (or mamba) installed and run:

```bash
conda env create -f environment.yml
```

Note that you do not need to have this set up if you are _only_ submitting jobs to the cluster.

## Downloading data

We requires some data files not packaged on Git due to their large size. These are required to be downloaded locally even
if you are running this on Singularity (as they are uploaded). To download these, do the following:

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
./download_cath.sh
```

## Tests
Tests are implemented through a mixture of doctests and unittests. To run unittests, run:

```bash
python -m unittest -v
```

## Singularity/amulet
To run on singularity/amulet, first install amulet following the instructions at https://amulet-docs.azurewebsites.net/main/setup.html. This should leave you with a conda environment named `amlt8`. Note that this environment should be _separate_ from the environment that is required to actually run model. To run training on singularity, run:

```bash
conda activate amlt8  # Activate the conda env.
amlt run -y scripts/amlt.yaml -o results
```

Within this `amlt.yaml` file, the python command contains a pointer to a config json file. Edit the path indicated here to
use a different configuration for training.

Note rearding the structure of the `amlt.yaml` file: installing packages via conda is very slow on the Singularity cluster, so
we recreate the same set of packages installed via pip instead of relying on conda.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
