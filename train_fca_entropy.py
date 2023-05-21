import gym
import envs
import torch
import numpy as np
import pickle
from core.policy import TanhGaussianPolicyWithDiscriminator, TanhGaussianPolicyWithEmbedding
from core.replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv

from argparse import ArgumentParser
from imitation.fca import FCA, FCA_MINE, FCA_Entropy
from itertools import product
import h5py

import os
wandb_dir = '/tmp'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = '/tmp'
import wandb

import d4rl
import time

def preprocess_dataset_with_prev_actions(mdpfile, envtype, stacksize=1, partially_observable=False, action_history_len=2):
    
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
    action_history_len = configs['action_history_len']
    print(f'-- Preprocessing dataset... ({envtype}, {stacksize})')
    path = preprocess_dataset_with_prev_actions(dataset, envtype, stacksize, configs['partially_observable'], action_history_len=action_history_len)
    # datafile.close()
    
    train_data = data_select_num_transitions(path, configs['train_data_num'])
    valid_data = data_select_num_transitions(path, configs['valid_data_num'], start_idx=900000)
    
    replay_buffer = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        action_history_len=action_history_len
    )
    replay_buffer.add_path(train_data)

    replay_buffer_valid = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        action_history_len=action_history_len
    )
    replay_buffer_valid.add_path(valid_data)

    wandb.init(project='copycat_imitation_1123_D4RLv2',
            dir=wandb_dir,
            config=configs,
            settings=wandb.Settings(start_method="fork")
            )

    if 'FCA_Ent' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            # hidden_sizes=configs['layer_sizes'],
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,
        )
    
        fca_trainer = FCA_Entropy(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            mi_lr=configs['mi_lr'],
            lr=configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            embedding_dim = embedding_dim,
            stacksize = stacksize,
            wandb = wandb,
            info_bottleneck_loss_coef = configs['info_bottleneck_loss_coef'],
            reg_coef = configs['reg_coef'],
            action_history_len=action_history_len
        )

        fca_trainer.train(total_iteration=configs['total_iteration'], 
                          mine_steps = configs['mine_steps'],
                          batch_size = configs['batch_size'])

        
    elif 'FCA_MINE' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            # hidden_sizes=configs['layer_sizes'],
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,            
        )
    
        fca_trainer = FCA_MINE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            mi_lr=configs['mi_lr'],
            lr=configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            embedding_dim = embedding_dim,
            stacksize = stacksize,
            wandb = wandb,
            info_bottleneck_loss_coef = configs['info_bottleneck_loss_coef'],
            mine_loss_coef = configs['mine_loss_coef'],
            action_history_len=action_history_len
        )

        fca_trainer.train(total_iteration=configs['total_iteration'], 
                          mine_steps = configs['mine_steps'],
                          batch_size = configs['batch_size'])

        
    elif 'FCA_MINE' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            # hidden_sizes=configs['layer_sizes'],
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=embedding_dim,                            # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300            
            device=device,            
        )
    
        fca_trainer = FCA_MINE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            mi_lr=configs['mi_lr'],
            lr=configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            embedding_dim = embedding_dim,
            stacksize = stacksize,
            wandb = wandb,
            info_bottleneck_loss_coef = configs['info_bottleneck_loss_coef'],
            mine_loss_coef = configs['mine_loss_coef'],
            action_history_len=action_history_len
        )

        fca_trainer.train(total_iteration=configs['total_iteration'], 
                          mine_steps = configs['mine_steps'],
                          batch_size = configs['batch_size'])


    else:
        policy = TanhGaussianPolicyWithDiscriminator(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            # hidden_sizes=configs['layer_sizes'],
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=configs['layer_sizes'][1],                # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300
            disc_hidden_size=configs['layer_sizes'][2],             # 300
            device=device,
            policy_std=0.1,
            disc_std=0.1
        )
        
        best_policy = TanhGaussianPolicyWithDiscriminator(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],        # 300
            embedding_dim=configs['layer_sizes'][1],                # 100
            policy_hidden_size=configs['layer_sizes'][2],           # 300
            disc_hidden_size=configs['layer_sizes'][2],             # 300
            device=device,
            policy_std=0.1,
            disc_std=0.1
        )
    
        fca_trainer = FCA(
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
            adversarial_loss_coef=configs['adversarial_loss_coef'],
            info_bottleneck_loss_coef=configs['info_bottleneck_loss_coef']
        )

        fca_trainer.train(total_iteration=configs['total_iteration'],
                        mine_steps = configs['mine_steps'],
                        batch_size = configs['batch_size'])



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid

    time.sleep(pid) # use for unstable file system
    
    envlist = ['Walker2d', 'Hopper', 'Ant', 'HalfCheetah']   # 'HalfCheetah', 'Walker2d', 'Ant']
    stacksizelist = [4, 3, 2, 1]                                # MDP=0
    seedlist = [0, 1, 2]                               # ,3,4]
    ib_coef_list = [1e-3]
    reg_coef_list = [1e-4, 2.]
    mine_steps_list = [10]
    batch_size = 512

    envtype, stacksize, seed, ib_coef, reg_coef = \
            list(product(envlist, stacksizelist, seedlist, ib_coef_list, reg_coef_list))[pid]

    if stacksize == -1:    
        stacksize_dict = {
            'Walker2d':     4,
            'Hopper':       3,
            'HalfCheetah':  2,
            'Ant':          3
        }
        stacksize = stacksize_dict['envtype']


    if ib_coef > 0:
        if reg_coef > 0:
            algorithm = f'FCA_EntREG_W{stacksize}_IB_reg{reg_coef}_{batch_size}'
        
    else:
        if reg_coef > 0:
            algorithm = f'FCA_EntREG_W{stacksize}_reg{reg_coef}_{batch_size}'


    if stacksize <=1:
        action_history_len = 1
    else:
        action_history_len = stacksize - 1

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

    configs = dict(
        algorithm=algorithm,
        layer_sizes=[300, 100, 300],
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_data_num=train_data_num,
        valid_data_num=2000,
        lr=3e-4,
        mi_lr=1e-4,
        envtype=envtype_lower,
        d4rl_env_name=d4rl_env_name,
        envname=envname,
        stacksize=stacksize,
        pid=pid,
        save_policy_path=None,   # not save when equals to None
        seed=seed,
        total_iteration=5e5,
        partially_observable=partially_observable,
        use_discriminator_action_input=True,
        info_bottleneck_loss_coef=ib_coef,
        mine_loss_coef=1e-4,
        adversarial_loss_coef=2.,
        discriminator_steps=1,
        action_history_len=action_history_len,
        mine_steps=1,
        batch_size=batch_size,
        reg_coef=reg_coef
    )

    configs['traj_load_path'] = traj_load_path
    configs['save_policy_path'] = f'results/{envname}/{algorithm}/num_train{train_data_num}/stack{stacksize}/seed{seed}'
    
    print(configs)

    train(configs)
