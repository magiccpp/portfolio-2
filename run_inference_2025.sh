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

python ./inference.py --period 4 &
PID7=$! # Store the process ID of the last background command

python ./inference.py --period 2 &
PID8=$! # Store the process ID of the last background command

python ./inference.py --period 1 &
PID9=$! # Store the process ID of the last background command

# Wait for all the parallel processes to finish
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6
wait $PID7
wait $PID8
wait $PID9


# Once all parallel tasks are done, run this command

python ./multi_horizon.py --periods 1,2,4,8,16,32,64,128 --output multi_shorter_horizon >> /tmp/multi_shorter_horizon.log 2>&1

# check current weekday, run below if it is Monday
python ./multi_horizon.py --periods 8,16,32,64,128,256 --output multi_horizon >> /tmp/multi_horizon.log 2>&1


# get the latest file under multi_horizon_short
latest_file=$(ls -t /home/ken/git/portfolio-2/multi_horizon_short/* | head -n 1)
(   echo "Subject: Janus: Valar Morghulis, Valar Dohaeris.";   echo "";   python verify_weights.py $latest_file; ) | msmtp xiaodong.ken.dai@gmail.com
(   echo "Subject: Janus: Valar Morghulis, Valar Dohaeris.";   echo "";   python verify_weights.py $latest_file; ) | msmtp w406971526@gmail.com
(   echo "Subject: Janus: Valar Morghulis, Valar Dohaeris.";   echo "";   python verify_weights.py $latest_file; ) | msmtp xulilin20081@gmail.com
# Deactivate the conda environment
conda deactivate


