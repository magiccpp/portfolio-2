#!/bin/bash
# Path to your conda initialization script
source /home/ken/anaconda3/etc/profile.d/conda.sh
# Activate the desired conda environment
conda activate stock
# Run your Python script
cd /home/ken/git/portfolio-2
# Run commands in parallel by sending them to the background
python ./inference.py --period 128 &
PID1=$! # Store the process ID of the last background command

python ./inference.py --period 256 &
PID2=$! # Store the process ID of the last background command

python ./inference.py --period 64 &
PID3=$! # Store the process ID of the last background command

python ./inference.py --period 32 &
PID4=$! # Store the process ID of the last background command

python ./inference.py --period 16 &
PID5=$! # Store the process ID of the last background command

python ./inference.py --period 8 &
PID6=$! # Store the process ID of the last background command


# Wait for all the parallel processes to finish
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6

# Once all parallel tasks are done, run this command
python ./multi_horizon.py --periods 8,16,32,64,128,256

# Deactivate the conda environment
conda deactivate


