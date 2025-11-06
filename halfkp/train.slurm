#!/bin/bash
#SBATCH --job-name=halfkp-train
#SBATCH --output=logs/halfkp-train-%j.log
#SBATCH --error=logs/halfkp-train-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Set environment
set -e
cd "$(dirname "$0")"

# Activate virtual environment
source ../.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with data directory argument
# Default to ../data/out if not specified
DATA_DIR="${1:-../data/out}"

echo "Starting HalfKP training job"
echo "Data directory: $DATA_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L)"

python train.py "$DATA_DIR"

echo "Training job completed at $(date)"
