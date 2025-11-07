#!/bin/bash
#SBATCH --job-name=data-download
#SBATCH --output=logs/data-download-%j.log
#SBATCH --error=logs/data-download-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared

echo "Starting data download job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

# Load Python module if available
module load python 2>/dev/null || true

# Install dependencies to user site-packages
python -m pip install --user requests zstandard

# Run download script
python download.py

echo "Data download job completed at $(date)"
