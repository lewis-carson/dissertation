#!/bin/bash
#SBATCH --job-name=data-download
#SBATCH --output=logs/data-download-%j.log
#SBATCH --error=logs/data-download-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Set environment
set -e
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting data download job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

# Install dependencies if needed
source .venv/bin/activate
pip install requests zstandard

# Run download script
python download.py

echo "Data download job completed at $(date)"
