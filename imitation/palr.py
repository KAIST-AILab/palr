import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

from imitation.bc import copy_nn_module
from core.hscic import estimate_hscic
                
class PALR(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True,
                 reg_coef=0.01, ridge_lambda=1e-3):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(PALR, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize
            
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)        
        
        self.num_eval_iteration = 50
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path
        
        # HSCIC Hyperparameters
        self.reg_coef = reg_coef
        self.ridge_lambda = ridge_lambda
        
        # For standardization
        self.standardize = standardize

        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std
                    

    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024, num_valid=2000):
        
        min_loss = 100000.
        max_score = -100000.
        
        batch_valid = self.replay_buffer_valid.get_batch(num_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']                
        actions_valid = batch_valid['actions'][:, -self.action_dim:]
        prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim]
                
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)

            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            prev_expert_action = batch['actions'][:, :-self.action_dim]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)

            neg_likelihood = - self.policy.log_prob(obs, actions).mean()            
            policy_action = self.policy(obs).rsample()
            
            if self.reg_coef != 0:                
                policy_embedding = self.policy.forward_embedding(obs)
                if self.standardize:
                    Y_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                    Z_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                    Y_std = Y_std.to(torch.float32)
                    Z_std = Z_std.to(torch.float32)
                else:
                    Y_std = prev_expert_action
                    Z_std = actions
                
                hscic_estimate = estimate_hscic(X=policy_embedding, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                
            else:
                hscic_estimate = 0.
            
            train_loss = neg_likelihood + self.reg_coef * hscic_estimate 

            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num+1) % eval_freq == 0:
                policy_action = self.policy(obs).sample()
                policy_action_valid = self.policy(obs_valid).sample()
                
                # Train data HSCIC (for debugging)                    
                policy_embedding = self.policy.forward_embedding(obs)
                if self.standardize:
                    Y_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                    Z_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]
                    p_std = (policy_action - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                    Y_std = Y_std.to(torch.float32)
                    Z_std = Z_std.to(torch.float32)
                    p_std = p_std.to(torch.float32)
                    
                else:
                    Y_std = prev_expert_action
                    Z_std = actions
                    p_std = policy_action
            
                hscic_estimate = estimate_hscic(X=policy_embedding, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                
                policy_embedding_valid = self.policy.forward_embedding(obs_valid)
                if self.standardize:
                    Y_std = (prev_expert_action_valid - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                    Z_std = (actions_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                    Y_std = Y_std.to(torch.float32)
                    Z_std = Z_std.to(torch.float32)
                else:
                    Y_std = prev_expert_action_valid
                    Z_std = actions_valid
                    p_std = policy_action
                    
                valid_hscic_estimate = estimate_hscic(X=policy_embedding_valid, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)                
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_expert_action_valid, Z=actions_valid, ridge_lambda=self.ridge_lambda)

                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()
                valid_loss = valid_neg_likelihood + self.reg_coef * valid_hscic_estimate

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={train_loss.item():.2f}, val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}',)
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({'train_total_loss':                  train_loss.item(), 
                                    'valid_total_loss':                  valid_loss.item(),
                                    'train_neg_likelihood':              neg_likelihood.item(),
                                    'valid_neg_likelihood':              valid_neg_likelihood.item(),
                                    'train_mean_hscic(rep,prev|target)': hscic_estimate,
                                    'valid_mean_hscic(rep,prev|target)': valid_hscic_estimate,
                                    'valid_mean_hscic(act,prev|target)': valid_hscic_estimate_action,
                                    'eval_episode_return':               eval_ret_mean
                                    }, step=num+1)

                if eval_ret_mean > max_score:
                    print(f'** max score record! ')
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
            
            while not done and t < maxtimestep:
                if self.standardize:
                    obs = (obs - self.obs_mean[0]) / self.obs_std[0]
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()
                
                next_obs, rew, done, _ = self.env.step(action)
                ret += rew
                
                obs_ = next_obs 
                obs_list.append(obs_)

                if len(obs_list) < self.stacksize:
                    obs_ = np.concatenate(obs_list)
                    obs = np.zeros(self.obs_dim * self.stacksize)
                    obs[-(len(obs_list)) * self.obs_dim:] = obs_
                
                else:
                    obs = np.concatenate(obs_list[-self.stacksize:])
                    
                t += 1
            
            rets.append(ret)
        
        return np.mean(rets), np.std(rets)
    
