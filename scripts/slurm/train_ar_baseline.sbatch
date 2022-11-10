#!/bin/bash
#SBATCH --partition=jamesz
#SBATCH --job-name=training               # Job name
#SBATCH --mail-type=FAIL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wukevin@stanford.edu  # Where to send mail	
#SBATCH --nodes=1                         # Run all processes on a single node
#SBATCH --ntasks=20                        # Number of processes
#SBATCH --mem=40gb                        # Job memory request
#SBATCH -G 2
#SBATCH -C GPU_SKU:RTX_2080Ti
#SBATCH --time=6-23:59:59                   # Time limit
#SBATCH --output=training_%j.log          # Standard output and error log
. /home/groups/jamesz/miniconda3/etc/profile.d/conda.sh
. ~/.bashrc
# Activate the appropriate conda environment
conda activate /home/groups/jamesz/wukevin/envs/protdiff
python /home/groups/jamesz/wukevin/projects/protdiff/bin/train_autoregressive.py
