#!/bin/bash
# Path to your conda initialization script
source /home/ken/anaconda3/etc/profile.d/conda.sh
# Activate the desired conda environment
conda activate stock
# Run your Python script
cd /home/ken/git/portfolio-2
python ./inference.py  --period 128
# Deactivate the conda environment
conda deactivate


