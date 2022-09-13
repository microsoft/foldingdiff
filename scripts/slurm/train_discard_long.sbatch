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
# Change to a directory for model to be output to
cd /home/groups/jamesz/wukevin/projects/protdiff_models/full_angles_discard_long
python /home/groups/jamesz/wukevin/projects/protdiff/bin/train.py /home/groups/jamesz/wukevin/projects/protdiff/config_jsons/cath_full_angles_discard_long.json --dryrun
