import gym
import envs
import torch
import numpy as np
import pickle
from core.policy import TanhGaussianPolicy
from core.replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv

from argparse import ArgumentParser
from imitation.bc import BC, BCwithMIReg
from imitation.valuedice import ValueDICE
from itertools import product
import h5py

import os
wandb_dir = '/tmp'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = '/tmp'
import wandb

import d4rl
import time

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

def preprocess_dataset_with_prev_actions(mdpfile, envtype, stacksize=1, partially_observable=False, action_history_len=None, train_with_action_history=False):
    
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

    if not action_history_len:
        action_history_len = max(stacksize-1, 1)
    
    terminals = np.array(mdpfile['terminals'])
    timeouts = np.array(mdpfile['timeouts'])
    rewards = np.array(mdpfile['rewards'])
    actions = np.array(mdpfile['actions'])

    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    n_data = observations.shape[0]
    new_observations_list = []
    new_next_observations_list = []
    prev_action_list = []
    action_history_list = []
    
    idx_from_initial_state = 0
    num_trajs = 0

    for i in range(n_data):
        if idx_from_initial_state == 0:
            prev_action = np.zeros(action_dim)
        else:
            prev_action = actions[i-1]
        prev_action_list.append(prev_action)

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

        if idx_from_initial_state < action_history_len:
            action_history = np.zeros(action_dim * action_history_len)
            action_history_ = np.concatenate(actions[i-idx_from_initial_state: i+1])
            action_history[-(idx_from_initial_state+1) * action_dim:] = action_history_

        else:
            action_history = np.concatenate(actions[i+1-action_history_len:i+1])

        new_observations_list.append(new_observation)
        new_next_observations_list.append(new_next_observation)
        action_history_list.append(action_history)

        idx_from_initial_state += 1
        if terminals[i] or timeouts[i]:
            idx_from_initial_state = 0
            num_trajs += 1

    new_observations = np.array(new_observations_list)
    new_next_observations = np.array(new_next_observations_list)

    # prev_actions = np.array(prev_action_list)
    action_histories = np.array(action_history_list)
    new_actions = np.concatenate((actions, action_histories), -1) 

    new_observations = np.concatenate((new_observations, action_histories), axis=-1)
    new_next_observations = np.concatenate((new_next_observations, np.zeros_like(action_histories)), axis=-1) # dummy prev actions

    new_paths = {
        'observations': new_observations,
        'next_observations': new_next_observations,
        'rewards': rewards,
        'actions': new_actions,
        'terminals': terminals,
        'timeouts': timeouts,
        'action_histories': action_histories
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
    
    d4rl_env = gym.make(configs['d4rl_env_name'])
    
    stacksize = configs['stacksize']
    if stacksize == 0:
        stacksize = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    envname, envtype = configs['envname'], configs['envtype']
    
    # with open(configs['traj_load_path'], 'rb') as f:
    #     path = pickle.load(f)
    traj_load_path = configs['traj_load_path']
    print(f'-- Loading dataset from {traj_load_path}...')
    dataset = d4rl_env.get_dataset()
    # datafile = h5py.File(traj_load_path, 'r')
    print(f'-- Done!')
    
    # if configs['partially_observable']:
    print(f'-- Preprocessing dataset... ({envtype}, {stacksize})')

    if 'MIReg' in configs['algorithm']:
        path = preprocess_dataset_with_prev_actions(dataset, envtype, stacksize, configs['partially_observable'], action_history_len=3)
        action_history_len = 3
    elif configs['train_with_action_history']:
        path = preprocess_dataset_with_prev_actions(dataset, envtype, stacksize, configs['partially_observable'], train_with_action_history=configs['train_with_action_history'])
        action_history_len = max(1, stacksize-1)
    else:
        path = preprocess_dataset(dataset, envtype, stacksize, configs['partially_observable'])
        action_history_len = 1
    # datafile.close()
    
    train_data = data_select_num_transitions(path, configs['train_data_num'])
    valid_data = data_select_num_transitions(path, configs['valid_data_num'], start_idx=900000)
    
    replay_buffer = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        train_with_action_history=configs['train_with_action_history'],
        action_history_len=action_history_len
    )
    replay_buffer.add_path(train_data)

    replay_buffer_valid = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        train_with_action_history=configs['train_with_action_history'],
        action_history_len=action_history_len
    )
    replay_buffer_valid.add_path(valid_data)

    wandb.init(project='copycat_imitation_1123_D4RLv2',
            dir=wandb_dir,
            config=configs,
            settings=wandb.Settings(start_method="fork")
            )

    if configs['train_with_action_history']:
        policy_input_dim = obs_dim * stacksize + action_dim * (stacksize-1)
    else:
        policy_input_dim = obs_dim * stacksize

    policy = TanhGaussianPolicy(
        obs_dim=policy_input_dim,
        action_dim=action_dim,
        hidden_sizes=configs['layer_sizes'],
        device=device,
    )
    
    best_policy = TanhGaussianPolicy(
        obs_dim=policy_input_dim,
        action_dim=action_dim,
        hidden_sizes=configs['layer_sizes'],
        device=device,
    )

    if 'ValueDICE' in configs['algorithm']:
        trainer = ValueDICE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device=device,
            envname = envname,
            actor_lr=configs['actor_lr'],
            nu_lr=configs['nu_lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim= action_dim,
            stacksize = stacksize,
            wandb=wandb,
            discount=configs['discount'],
            grad_reg_coef=configs['grad_reg_coef']
        )
        trainer.train(total_iteration=configs['total_iteration'], batch_size=configs['batch_size'])

    elif 'BC' in configs['algorithm']:
        trainer = BC(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device=device,
            envname = envname,
            lr=configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim=action_dim,
            stacksize = stacksize,
            wandb=wandb,
            l2_reg_coef=configs['l2_reg_coef'],
            train_with_action_history=True
        )
        trainer.train(total_iteration=configs['total_iteration'], batch_size=configs['batch_size'])

    elif 'MIRegAlter' in configs['algorithm']:
        trainer = BCwithMIReg(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device=device,
            envname = envname,
            lr=configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim=action_dim,
            stacksize = stacksize,
            wandb=wandb,
            mi_reg_coef=configs['mi_reg_coef'],
            action_history_len=action_history_len
        )
        trainer.train(total_iteration=configs['total_iteration'], batch_size=configs['batch_size'], alternate=True)

    elif 'MIReg' in configs['algorithm']:
        trainer = BCwithMIReg(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device=device,
            envname = envname,
            lr=configs['lr'],
            mi_lr=configs['mi_lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim=action_dim,
            stacksize = stacksize,
            wandb=wandb
        )
        trainer.train(total_iteration=configs['total_iteration'], batch_size=configs['batch_size'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid

    time.sleep(pid) # use for unstable file system
    
    envlist = ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant']  #['Hopper', 'Walker2d', 'HalfCheetah', 'Ant']
    stacksizelist = [1,2,3,4] #1,2,3,4] #,3,4,0,1]          # MDP
    seedlist = [0,1,2]

    mi_lr_list = [1e-4]
    mi_reg_coef_list = [1e-5]

    envtype, stacksize, seed, mi_lr, mi_reg_coef = list(product(envlist, stacksizelist, seedlist, mi_lr_list, mi_reg_coef_list))[pid]

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
    d4rl_env_name = f'{envtype_lower}-expert-v2'

    train_data_num = 20000
    # algorithm='BC_L2Reg_v2'
    algorithm='BC-OAH'

    configs = dict(
        algorithm=algorithm,         # 'BC', 'ValueDICE'
        layer_sizes=[300, 100, 300],
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_data_num=train_data_num,
        valid_data_num=2000,
        lr=3e-4,
        mi_lr=mi_lr,
        envtype=envtype_lower,
        d4rl_env_name=d4rl_env_name,
        envname=envname,
        stacksize=stacksize,
        pid=pid,
        save_policy_path=None,   # not save when equals to None
        seed=seed,
        total_iteration=5e5,
        partially_observable=partially_observable,
        # ValueDICE params
        # discount=0.99,
        # actor_lr=1e-5,
        # nu_lr=1e-3,
        batch_size=256,
        l2_reg_coef=mi_reg_coef, # 0.01
        mi_reg_coef=mi_reg_coef,
        train_with_action_history = True
        # grad_reg_coef=10.
    )

    configs['traj_load_path'] = traj_load_path
    configs['save_policy_path'] = f'results/{algorithm}/{envname}/num_train{train_data_num}/stack{stacksize}/seed{seed}'

    print(configs)

    train(configs)
