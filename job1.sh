#!/bin/bash
#SBATCH --job-name=minesweeper_python_job   # Descriptive job name
#SBATCH --account=PHS0378
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1                 # Number of tasks per node
#SBATCH --cpus-per-task=1          # Number of CPUs per task
#SBATCH --mem=4G                   # Memory per node
#SBATCH --time=09:00:00            # Time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1
# Load necessary modules (if applicable)
module load python/3.8  # Example module load
module load cuda/11.8.0
# Activate virtual environment (if applicable)
# source activate my_env  # Example activation
source activate games38
# Execute your Python script
python MineLearnMask.py
