#!/bin/bash
#SBATCH --job-name=halfkp-train
#SBATCH --output=logs/halfkp-train-%j.log
#SBATCH --error=logs/halfkp-train-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=ug-gpu-small

# Set environment
set -e
cd "$(dirname "$0")"

# Load Python module if available
module load python 2>/dev/null || true

# Install dependencies to user site-packages
python -m pip install --user torch wandb

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with data directory argument
# Default to ./data/out if not specified
DATA_DIR="${1:-./data/}"

echo "Starting HalfKP training job"
echo "Data directory: $DATA_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L)"

python train.py "$DATA_DIR"

echo "Training job completed at $(date)"
