#!/bin/bash
#SBATCH --partition=jamesz
#SBATCH --job-name=omegafold               # Job name
#SBATCH --mail-type=ALL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wukevin@stanford.edu  # Where to send mail	
#SBATCH --nodes=1                         # Run all processes on a single node
#SBATCH --ntasks=10                       # Number of processes
#SBATCH --mem=40gb                        # Job memory request
#SBATCH -G 2
#SBATCH -C GPU_SKU:RTX_2080Ti
#SBATCH --time=6-23:59:59                  # Time limit
#SBATCH --output=omegafold_%j.log          # Standard output and error log
. /home/groups/jamesz/miniconda3/etc/profile.d/conda.sh
. ~/.bashrc
# Activate the appropriate conda environment
conda activate /home/groups/jamesz/wukevin/envs/omegafold
python /home/groups/jamesz/wukevin/projects/protdiff/bin/omegafold_across_gpus.py proteinmpnn_residues/*.fasta -o omegafold_predictions_proteinmpnn --weights /home/groups/jamesz/wukevin/software/omegafold/release1.pt
