#!/bin/bash
for i in {0..4}
do
tmux new -d -s session-$i "LD_LIBRARY_PATH=/home/syseo/.mujoco/mujoco210/bin:/usr/lib/nvidia /home/syseo/anaconda3/envs/copycat/bin/python train_bcoh_from_mdpdata.py --pid=$i" &
done