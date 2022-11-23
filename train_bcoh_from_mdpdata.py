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
wandb_dir = '/tmp'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = '/tmp'
import wandb

import d4rl

def preprocess_dataset(mdpfile, envtype, stacksize=1, partially_observable=False):
    
    indx = list(np.arange(20))
    # Indices of position information observations
    if partially_observable:
        envtype_to_idx = {
            'hopper': indx[:5], 
            'ant': indx[:13], 
            'walker2d': indx[:8], 
            'halfcheetah': indx[:4] + indx[8:13]
        }
        obs_idx = envtype_to_idx[envtype]
        observations = np.array(mdpfile['observations'])[:, obs_idx]
        next_observations = np.array(mdpfile['next_observations'])[:, obs_idx]
    else:
        observations = np.array(mdpfile['observations'])
        next_observations = np.array(mdpfile['next_observations'])
    
    new_path = {}
    
    done = False
    
    terminals = np.array(mdpfile['terminals'])
    timeouts = np.array(mdpfile['timeouts'])
    rewards = np.array(mdpfile['rewards'])
    actions = np.array(mdpfile['actions'])

    obs_dim = observations.shape[-1]
    n_data = observations.shape[0]
    new_observations_list = []
    new_next_observations_list = []
    
    idx_from_initial_state = 0
    num_trajs = 0

    for i in range(n_data):
        if idx_from_initial_state < stacksize:
            if idx_from_initial_state == 0:
                initial_obs = observations[i]
            
            new_observation = np.zeros(obs_dim * stacksize)
            new_observation_ = np.concatenate(observations[i-idx_from_initial_state: i+1])
            new_observation[-(idx_from_initial_state+1) * obs_dim:] = new_observation_
            
            new_next_observation = np.zeros(obs_dim * stacksize)
            new_next_observation_ = np.concatenate(next_observations[i-idx_from_initial_state: i+1])
            new_next_observation[-(idx_from_initial_state+1) * obs_dim:] = new_next_observation_
            
            if idx_from_initial_state + 1 != stacksize:
                new_next_observation[-(idx_from_initial_state+2) * obs_dim:-(idx_from_initial_state+1) * obs_dim] \
                    = initial_obs
            
        else:
            new_observation = np.concatenate(observations[i+1-stacksize:i+1])
            new_next_observation = np.concatenate(next_observations[i+1-stacksize:i+1])

        new_observations_list.append(new_observation)
        new_next_observations_list.append(new_next_observation)

        idx_from_initial_state += 1
        if terminals[i] or timeouts[i]:
            idx_from_initial_state = 0
            num_trajs += 1

    new_observations = np.array(new_observations_list)
    new_next_observations = np.array(new_next_observations_list)

    new_paths = {
        'observations': new_observations,
        'next_observations': new_next_observations,
        'rewards': rewards,
        'actions': actions,
        'terminals': terminals,
        'timeouts': timeouts
    }
    
    return new_paths

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
    
    d4rl_env = env.make(configs['d4rl_env_name'])
    
    stacksize = configs['stacksize']
    if stacksize == 0:
        stacksize = 1
    
    envname, envtype = configs['envname'], configs['envtype']
    
    # with open(configs['traj_load_path'], 'rb') as f:
    #     path = pickle.load(f)
    traj_load_path = configs['traj_load_path']
    print(f'-- Loading dataset from {traj_load_path}...')
    dataset = d4rl.get_dataset()
    # datafile = h5py.File(traj_load_path, 'r')
    print(f'-- Done!')
    
    # if configs['partially_observable']:
    print(f'-- Preprocessing dataset... ({envtype}, {stacksize})')
    path = preprocess_dataset(dataset, envtype, stacksize, configs['partially_observable'])
    datafile.close()
    
    train_data = data_select_num_transitions(path, configs['train_data_num'])
    valid_data = data_select_num_transitions(path, configs['valid_data_num'], start_idx=900000)
    
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

    wandb.init(project='copycat_imitation_1122_D4RLv2',
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
        device='cuda'
    )
    
    best_policy = TanhGaussianPolicy(
        obs_dim=obs_dim * stacksize,
        action_dim=action_dim,
        # min_log_std=-3.67,      # log 0.25
        # max_log_std= 0.,        # log 1
        hidden_sizes=configs['layer_sizes'],
        device='cuda'
    )
    
    bc_trainer = BC(
        policy = policy,
        best_policy = best_policy,
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
    
    envlist = ['HalfCheetah', 'Hopper', 'HalfCheetah', 'Ant']
    stacksizelist = [0,1,2,3,4] # MDP
    seedlist = [0,1,2]

    envtype, stacksize, seed = list(product(envlist, stacksizelist, seedlist))[pid]

    if stacksize == 0 :
        # MDP
        partially_observable = False
        envname = f'{envtype}-v2'
        
    else:
        # POMDP
        partially_observable = True
        envname = f'PO{envtype}-v0'
        
    envtype_lower = envtype.lower()
    traj_load_path = f'/tmp/{envtype_lower}_expert-v2.hdf5'
    d4rl_env_name = f'{envtype_lower}_expert-v2'

    configs = dict(
        layer_sizes=[300, 100, 300],
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_data_num=5000,
        valid_data_num=1000,
        lr=3e-4,
        envtype=envtype_lower,
        d4rl_env_name=d4rl_env_name,
        envname=envname,
        stacksize=stacksize,
        pid=pid,
        save_policy_path=None,   # not save when equals to None
        seed=seed,
        total_iteration=5e5,
        partially_observable=partially_observable
    )

    configs['traj_load_path'] = traj_load_path
    configs['save_policy_path'] = f'results/{envname}/stack{stacksize}/seed{seed}'

    print(configs)

    train(configs)
