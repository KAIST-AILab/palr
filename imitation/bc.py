import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class BC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, l2_reg_coef=0., train_with_action_history=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
    
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.device = device
        self.num_eval_iteration = 50
        self.envname = envname
        self.l2_reg_coef = l2_reg_coef
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize
        self.stacksize_action = max(stacksize-1, 1)
        self.train_with_action_history = train_with_action_history

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            actions = batch['actions']
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            actions = actions[:, :self.action_dim]

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer          
            neg_likelihood = -self.policy.log_prob(obs, actions).mean()
            l2_norm = sum((p ** 2).sum() for p in self.policy.parameters())
            loss = neg_likelihood + self.l2_reg_coef * 0.5 * l2_norm
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (num+1) % iter_per_valid == 0:

                batch_valid = self.replay_buffer_valid.random_batch(batch_size)
                obs_valid = batch_valid['observations']
                actions_valid = batch_valid['actions']
                        
                obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                actions_valid = actions_valid[:, :self.action_dim]
                                         
                valid_loss = -self.policy.log_prob(obs_valid, actions_valid).mean()
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                
                if self.wandb:
                    self.wandb.log({'BC_loss_train': loss, 
                                    'BC_loss_valid': valid_loss,
                                    'eval_episode_return': eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** min val_loss! ')
                    # min_loss = valid_loss
                    # policy.state_dict()
                    max_score = eval_ret_mean
                    copy_nn_module(self.policy, self.best_policy)
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
                    
                    
    def evaluate(self, num_iteration=5):
        rets = []
        maxtimestep = 1000
        for num in range(0, num_iteration):
            obs_list = []

            obs = np.zeros(self.obs_dim * self.stacksize)            
            
            obs_ = self.env.reset()
            obs_list.append(obs_)

            obs = np.zeros(self.obs_dim * self.stacksize)
            obs[- self.obs_dim:] = obs_

            done = False
            t = 0
            ret = 0.

            if self.train_with_action_history:
                prev_actions_list = []
                prev_actions = np.zeros(self.action_dim * (self.stacksize - 1))                
            
            while not done and t < maxtimestep:
                # obs = obs[:true_obs_dim]
                if self.train_with_action_history:
                    obs = np.concatenate((obs, prev_actions), -1)
                    
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()

                next_obs, rew, done, env_info = self.env.step(action)

                ret += rew
                
                obs_ = next_obs 
                obs_list.append(obs_)

                if len(obs_list) < self.stacksize:
                    obs_ = np.concatenate(obs_list)
                    obs  = np.zeros(self.obs_dim * self.stacksize)
                    obs[-(len(obs_list)) * self.obs_dim:] = obs_
                
                else:
                    obs = np.concatenate(obs_list[-self.stacksize:])

                if self.train_with_action_history:
                    prev_actions_list.append(action)

                    if len(prev_actions_list) < self.stacksize_action:
                        prev_actions_ = np.concatenate(prev_actions_list)
                        prev_actions = np.zeros(self.action_dim * (self.stacksize_action))
                        prev_actions[-(len(prev_actions_list)) * self.action_dim:] = prev_actions_
                    
                    else:
                        prev_actions = np.concatenate(prev_actions_list[-(self.stacksize_action):])
                    
                t += 1
            
            rets.append(ret)
        
        return np.mean(rets), np.std(rets) / np.sqrt(len(rets))
    
    
    def evaluate_with_copycat_ratio(self, num_iteration=5, copycat_ratio=0.):
        # stochastic evaluation
        rets = []
        maxtimestep = 1000
        for num in range(0, num_iteration):
            obs_list = []

            obs = np.zeros(self.obs_dim * self.stacksize)            
            
            obs_ = self.env.reset()
            obs_list.append(obs_)

            obs = np.zeros(self.obs_dim * self.stacksize)
            obs[- self.obs_dim:] = obs_
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            prev_obs = obs

            done = False
            t = 0
            ret = 0.

            if self.train_with_action_history:
                prev_actions_list = []
                prev_actions = np.zeros(self.action_dim * (self.stacksize - 1))                
            
            while not done and t < maxtimestep:
                # obs = obs[:true_obs_dim]
                # if self.train_with_action_history:
                #     obs = np.concatenate((obs, prev_actions), -1)
                    

                # -- deterministic
                # action = self.policy(obs).mean.cpu().detach().numpy()
                # -- stochastic
                prob = random.random()
                
                if prob > copycat_ratio:
                    action = self.policy(obs).mean.cpu().detach().numpy()
                else:
                    action = self.policy(prev_obs).mean.cpu().detach().numpy()

                next_obs, rew, done, env_info = self.env.step(action)

                ret += rew
                
                prev_obs = obs
                obs_ = next_obs 
                obs_list.append(obs_)

                if len(obs_list) < self.stacksize:
                    obs_ = np.concatenate(obs_list)
                    obs  = np.zeros(self.obs_dim * self.stacksize)
                    obs[-(len(obs_list)) * self.obs_dim:] = obs_
                
                else:
                    obs = np.concatenate(obs_list[-self.stacksize:])
                    
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

                if self.train_with_action_history:
                    prev_actions_list.append(action)

                    if len(prev_actions_list) < self.stacksize_action:
                        prev_actions_ = np.concatenate(prev_actions_list)
                        prev_actions = np.zeros(self.action_dim * (self.stacksize_action))
                        prev_actions[-(len(prev_actions_list)) * self.action_dim:] = prev_actions_
                    
                    else:
                        prev_actions = np.concatenate(prev_actions_list[-(self.stacksize_action):])
                    
                t += 1
            
            rets.append(ret)
        
        return np.mean(rets), np.std(rets)


    def evaluate_and_collect_data(self, num_iteration=5):
        rets = []
        maxtimestep = 1000

        data_dict = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'true_states': [],
            'observations_only': [], # without action
            'dones': [],

        }

        for num in range(0, num_iteration):
            obs_list = []

            obs = np.zeros(self.obs_dim * self.stacksize)            
            
            obs_ = self.env.reset()
            obs_list.append(obs_)

            obs = np.zeros(self.obs_dim * self.stacksize)
            obs[- self.obs_dim:] = obs_

            done = False
            t = 0
            ret = 0.

            if self.train_with_action_history:
                prev_actions_list = []
                prev_actions = np.zeros(self.action_dim * (self.stacksize - 1))                
            
            while not done and t < maxtimestep:
                # obs = obs[:true_obs_dim]    
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()

                next_obs, rew, done, env_info = self.env.step(action)

                ret += rew
                
                obs_ = next_obs 
                obs_list.append(obs_)

                if len(obs_list) < self.stacksize:
                    obs_ = np.concatenate(obs_list)
                    obs  = np.zeros(self.obs_dim * self.stacksize)
                    obs[-(len(obs_list)) * self.obs_dim:] = obs_
                
                else:
                    obs = np.concatenate(obs_list[-self.stacksize:])

                if self.train_with_action_history:
                    prev_actions_list.append(action)

                    if len(prev_actions_list) < self.stacksize_action:
                        prev_actions_ = np.concatenate(prev_actions_list)
                        prev_actions = np.zeros(self.action_dim * (self.stacksize_action))
                        prev_actions[-(len(prev_actions_list)) * self.action_dim:] = prev_actions_
                    
                    else:
                        prev_actions = np.concatenate(prev_actions_list[-(self.stacksize_action):])
                    
                t += 1
            
            rets.append(ret)
        
        return np.mean(rets), np.std(rets)



