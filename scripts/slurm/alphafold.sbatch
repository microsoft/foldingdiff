#!/bin/bash
#SBATCH --partition=jamesz
#SBATCH --job-name=alphafold               # Job name
#SBATCH --mail-type=ALL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wukevin@stanford.edu  # Where to send mail	
#SBATCH --nodes=1                         # Run all processes on a single node
#SBATCH --ntasks=4                       # Number of processes
#SBATCH --mem=16gb                        # Job memory request
#SBATCH -G 1
#SBATCH -C GPU_SKU:RTX_2080Ti
#SBATCH --time=6-23:59:59                  # Time limit
#SBATCH --output=alphafold_%j.log          # Standard output and error log

ml gcc/6.3.0
ml cuda/11.7.1
. /home/groups/jamesz/miniconda3/etc/profile.d/conda.sh
. ~/.bashrc

# Expects a3m files under "input" directory, outputs to "out" direcotry
# Auto handles environment
/scratch/users/wukevin/software/localcolabfold/colabfold_batch/bin/colabfold_batch inputs out --num-recycle 15 --num-models 1 --model-order 1 --sort-queries-by random
