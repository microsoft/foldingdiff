# Singularity/amulet

To run on singularity/amulet, make sure you have already downloaded the CATH dataset (see instructions above). If you do not have amulet installed, folow the instructions at <https://amulet-docs.azurewebsites.net/main/setup.html>. This should leave you with a conda environment named `amlt8`. Note that this environment should be _separate_ from the environment for the diffusion model itself. Note that you do _not_ need to create the given `environment.yml` to submit to amulet/singularity; the environment for running the code is separately set up within the Singularity compute cluster.

With these two requirements, to run training on singularity, run:

```bash
conda activate amlt8  # Activate the conda env.
amlt run -y scripts/amlt.yaml -o results
```

Within this `amlt.yaml` file, the python command contains a pointer to a config json file. Edit the path indicated here to
use a different configuration for training.

Note rearding the structure of the `amlt.yaml` file: installing packages via conda is very slow on the Singularity cluster, so
we recreate the same set of packages installed via pip instead of relying on conda.
