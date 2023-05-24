import os
wandb_dir = '.'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = './dataset/'
import wandb
import envs
import d4rl
import gym

import torch

from imitation.bc import BC
from imitation.rap import RAP
from imitation.fca import FCA
from imitation.mine import MINE_BC
from imitation.palr import PALR

from argparse import ArgumentParser
from itertools import product

from core.policy import TanhGaussianPolicyWithEmbedding, TanhGaussianRAPPolicy
from core.replay_buffer import EnvReplayBuffer
from core.preprocess import preprocess_dataset_with_prev_actions, data_select_num_transitions
from rlkit.envs.wrappers import NormalizedBoxEnv

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
    
    traj_load_path = configs['traj_load_path']
    print(f'-- Loading dataset from {traj_load_path}...')
    dataset = d4rl_env.get_dataset()
    print(f'-- Done!')
    
    print(f'-- Preprocessing dataset... ({envtype}, {stacksize})')
    path = preprocess_dataset_with_prev_actions(dataset, envtype, stacksize, configs['partially_observable'], action_history_len=2)    
    
    train_data = data_select_num_transitions(path, configs['train_data_num'])
    valid_data = data_select_num_transitions(path, configs['valid_data_num'], start_idx=900000)
    
    replay_buffer = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        action_history_len=2
    )
    replay_buffer.add_path(train_data)

    replay_buffer_valid = EnvReplayBuffer(
        configs['replay_buffer_size'],
        env,
        stacksize,
        action_history_len=2
    )
    replay_buffer_valid.add_path(valid_data)
    
    if configs['standardize']:
        obs_mean, obs_std, act_mean, act_std = replay_buffer.calculate_statistics()
        replay_buffer_valid.set_statistics(obs_mean, obs_std, act_mean, act_std)
        
    # to use wandb, initialize here, e.g.
    # wandb.init(project='palr', dir=wandb_dir, config=configs)
    wandb = None
    
    if 'BC' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,            
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            device=device            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            device=device
        )
        
        trainer = BC(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,            
            stacksize = stacksize,
            wandb = wandb,            
            standardize=configs['standardize']
        )

        trainer.train(total_iteration=configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'],
                      num_valid = configs['valid_data_num'])
        
    elif 'RAP' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianRAPPolicy(
            obs_dim=obs_dim,
            stack_size=stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            residual_hidden_size=configs['additional_network_size'],
            device=device,
        )
        
        best_policy = TanhGaussianRAPPolicy(
            obs_dim=obs_dim,
            stack_size=stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            residual_hidden_size=configs['additional_network_size'],
            device=device,
        )
    
        trainer = RAP(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,  
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,            
            embedding_dim = embedding_dim,
            stacksize = stacksize,
            wandb = wandb,                        
            standardize=configs['standardize']
        )

        trainer.train(total_iteration = configs['total_iteration'], 
                      eval_freq  = configs['eval_freq'],
                      batch_size = configs['batch_size'],
                      num_valid = configs['valid_data_num'])
        
    elif 'FCA' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,            
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],            
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],            
            device=device,
        )
    
        trainer = FCA(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            lr = configs['lr'],
            wandb = wandb,
            save_policy_path = configs['save_policy_path'],            
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            standardize=configs['standardize'],           
            embedding_dim = embedding_dim,
            entropy_hidden_size = configs['additional_network_size'],
            entropy_lr = configs['inner_lr'],
            reg_coef = configs['reg_coef'],
            info_bottleneck_loss_coef = configs['info_bottleneck_loss_coef']
        )

        trainer.train(total_iteration = configs['total_iteration'], 
                      eval_freq  = configs['eval_freq'],
                      batch_size = configs['batch_size'],
                      num_valid = configs['valid_data_num'],
                      inner_steps = configs['inner_steps'],)
        
    elif 'MINE' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,            
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],            
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],            
            device=device,            
        )
    
        trainer = MINE_BC(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            lr = configs['lr'],
            wandb = wandb,            
            save_policy_path = configs['save_policy_path'],            
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            embedding_dim = embedding_dim,
            standardize=configs['standardize'],            
            mine_lr = configs['inner_lr'],
            reg_coef = configs['reg_coef'],            
            info_bottleneck_loss_coef = configs['info_bottleneck_loss_coef']
        )

        trainer.train(total_iteration = configs['total_iteration'], 
                      eval_freq  = configs['eval_freq'],
                      inner_steps = configs['inner_steps'],
                      batch_size = configs['batch_size'],
                      num_valid = configs['valid_data_num'])
        
    elif 'PALR' in configs['algorithm']:
        embedding_dim = configs['layer_sizes'][1]
        policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,            
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            device=device,            
        )
        
        best_policy = TanhGaussianPolicyWithEmbedding(
            obs_dim=obs_dim * stacksize,
            action_dim=action_dim,
            embedding_hidden_size=configs['layer_sizes'][0],
            embedding_dim=embedding_dim,
            policy_hidden_size=configs['layer_sizes'][2],
            device=device,
        )
    
        trainer = PALR(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            wandb = wandb,
            reg_coef = configs['reg_coef'],
            ridge_lambda = configs['ridge_lambda'],
            standardize=configs['standardize']
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq  = configs['eval_freq'],
                      batch_size = configs['batch_size'],
                      num_valid  = configs['valid_data_num'])
    
    else: 
        raise NotImplementedError       

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=4, type=int)
    args = parser.parse_args()
    pid = args.pid
    
    # Hyperparameter Grid
    methodlist        = ['BC', 'RAP', 'FCA', 'MINE', 'PALR']           # candidates: 'BC', 'RAP', 'FCA', 'MINE', 'PALR'
    envlist           = ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant']   # candidates: 'Hopper', 'Walker2d', 'HalfCheetah', 'Ant'
    stacksizelist     = [2, 4]
    seedlist          = [0, 1, 2, 3, 4]
    reg_coef_list     = [10.]
    batch_size_list   = [1024]
    dataset_size_list = [30000]
    ridge_lambda_list = [1e-5]
    
    standardize = True    
    
    method, envtype, stacksize, seed, reg_coef, batch_size, dataset_size, ridge_lambda = \
        list(product(methodlist, envlist, stacksizelist, seedlist, reg_coef_list, batch_size_list, dataset_size_list, ridge_lambda_list))[pid]    
        
    if method == 'BC':
        reg_coef = 0.
    
    if method == 'FCA':
        ib_coef = 0.01
    else:
        ib_coef = 0.
        
    if method == 'FCA' or method == 'MINE':
        inner_steps = 5
        reg_coef = 0.1
    else:
        inner_steps = 1
        
    algorithm = f'{method}_W{stacksize}_corrected'

    if stacksize == 0 :        # MDP
        partially_observable = False
        envname = f'{envtype}-v2'        
    else:                      # POMDP
        partially_observable = True
        envname = f'PO{envtype}-v0'
        
    envtype_lower = envtype.lower()
    traj_load_path = f'/tmp/{envtype_lower}_expert-v2.hdf5'
    d4rl_env_name = f'{envtype_lower}-expert-v2'

    train_data_num = dataset_size

    configs = dict(
        algorithm=algorithm,        
        layer_sizes=[128, 64, 128],
        additional_network_size=128,
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_data_num=train_data_num,
        valid_data_num=2000,
        eval_freq=1000,
        lr=3e-4,
        inner_lr=1e-4,
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
        reg_coef=reg_coef,  
        inner_steps=inner_steps,
        batch_size=batch_size,
        ridge_lambda=ridge_lambda,
        standardize=standardize
    )

    configs['traj_load_path'] = traj_load_path
    configs['save_policy_path'] = f'results/{envname}/{algorithm}/alpha{reg_coef}/num_train{train_data_num}/stack{stacksize}/seed{seed}'
    
    print(configs)
    train(configs)
