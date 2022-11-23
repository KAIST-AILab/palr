#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'
# echo "hostname: `hostname`"
# echo "PATH=$PATH"
# echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export DISABLE_TQDM=TRUE
cd /ext2/syseo/copycat_imitation
D4RL_DATASET_DIR=/tmp /home/$USER/anaconda3/envs/copycat/bin/python train_bcoh.py --pid=$1