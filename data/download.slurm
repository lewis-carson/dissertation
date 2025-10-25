#!/bin/bash
#SBATCH --job-name=download_chess_data
#SBATCH --output=download_%j.log
#SBATCH --error=download_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB

# Download chess training data files
for idx in $(seq -f "%05g" 0 250)
do
  wget https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag
done
