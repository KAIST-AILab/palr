import gym
import envs
import torch
import numpy as np
import pickle
from core.policy import TanhGaussianPolicy
from core.replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv

from argparse import ArgumentParser
from imitation.bc import BC
from itertools import product
import h5py

import os
wandb_dir = '/tmp/syseo'
os.environ['WANDB_DIR'] = wandb_dir
import wandb

def data_select_num_transitions(path, num_transitions=1000, start_idx=0, random=False):
    new_path = {}
    
    if random:
        num_full_trajs = len(path['observations'])
        choice_idx = np.random.choice(num_full_trajs, num_transitions)
        
    else:
        choice_idx = np.arange(start_idx, start_idx + num_transitions)
        
    for key in path.keys():
        new_path[key] = np.array(path[key])[choice_idx]
        
    return new_path

def train(configs):
    env = NormalizedBoxEnv(gym.make(configs['envname']))
    obs_dim    = env.observation_space.low.size
    action_dim = env.action_space.low.size
    
    stacksize = configs['stacksize']
    if stacksize == 0:
        stacksize = 1
    
    envname = configs['envname']
    
    # with open(configs['traj_load_path'], 'rb') as f:
    #     path = pickle.load(f)

    datafile = h5py.File(configs['traj_load_path'], 'r')
    train_data = data_select_num_transitions(datafile, configs['train_data_num'])
    valid_data = data_select_num_transitions(datafile, configs['valid_data_num'], start_idx=900000)
    datafile.close()
    
    replay_buffer = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize
    )
    replay_buffer.add_path(train_data)

    replay_buffer_valid = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize
    )
    replay_buffer_valid.add_path(valid_data)

    wandb.init(project='copycat_imitation_1118',
            dir=wandb_dir,
            config=configs,
            settings=wandb.Settings(start_method="fork")
            )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim * stacksize,
        action_dim=action_dim,
        # min_log_std=-3.67,      # log 0.25
        # max_log_std= 0.,        # log 1
        hidden_sizes=configs['layer_sizes'],
    )
    
    bc_trainer = BC(
        policy = policy,
        env = env,
        replay_buffer = replay_buffer,
        replay_buffer_valid = replay_buffer_valid,
        seed = configs['seed'],
        device='cuda',
        envname = envname,
        lr=configs['lr'],
        save_policy_path = configs['save_policy_path'],
        obs_dim = obs_dim,
        stacksize = stacksize,
        wandb=wandb
    )

    bc_trainer.train(total_iteration=configs['total_iteration'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid
    
    envlist = ['Walker2d', 'HalfCheetah'] #, 'Hopper', 'HalfCheetah', 'Ant']
    stacksizelist = [0,1,2,3,4] # MDP
    seedlist = [0,1,2]

    envtype, stacksize, seed = list(product(envlist, stacksizelist, seedlist))[pid]

    if stacksize == 0 :
        # MDP
        envname = f'{envtype}-v2'
        envtype_lower = envtype.lower()
        traj_load_path = f'dataset/mdps/{envtype_lower}_expert.hdf5'
    else:
        # POMDP
        envname = f'PO{envtype}-v0'
        envtype_lower = envtype.lower()
        if stacksize > 1:
            traj_load_path = f'dataset/pomdps/{envtype_lower}_expert_h{stacksize}.hdf5'
        else:
            traj_load_path = f'dataset/pomdps/{envtype_lower}_expert.hdf5'

    configs = dict(
        layer_sizes=[300, 100, 300],
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_data_num=5000,
        valid_data_num=1000,
        lr=3e-4,
        envtype=envtype,
        envname=envname,
        stacksize=stacksize,
        pid=pid,
        save_policy_path=None,   # not save when equals to None
        seed=seed,
        total_iteration=3e6
    )

    configs['traj_load_path'] = traj_load_path

    print(configs)

    train(configs)
