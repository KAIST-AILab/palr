import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from imitation.bc import copy_nn_module
from core.hscic import estimate_hscic

class RAP(nn.Module):
    # Implementation of Residual Action Prediction (ECCV 2022)
    # - https://arxiv.org/pdf/2207.09705.pdf
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, embedding_dim=1, stacksize=1, standardize=False
                 ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(RAP, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
                
        self.device = device
       
        self.m_embedding_optimizer = optim.Adam(policy.history_embedding_params, lr=lr)
        self.h_embedding_optimizer = optim.Adam(policy.single_embedding_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        self.residual_optimizer = optim.Adam(policy.residual_params, lr=lr)

        self.num_eval_iteration = 50        
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.stacksize = stacksize
        
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
        
        max_score = -100000.       
        min_loss = 100000. 
        
        batch_valid = self.replay_buffer_valid.get_batch(num_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']
        actions_valid = batch_valid['actions'][:, -self.action_dim:]
        prev_actions_valid = batch_valid['actions'][:, :-self.action_dim]                
        
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        prev_actions_valid = torch.tensor(prev_actions_valid, dtype=torch.float32, device=self.device)        
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
            
            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            prev_actions = batch['actions'][:, :-self.action_dim]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)

            self.m_embedding_optimizer.zero_grad()
            self.residual_optimizer.zero_grad()         
            
            # m : history embedding, h : single observation embedding
            m, _ = self.policy.forward_embedding(obs)            
            action_residuals = actions - prev_actions
            action_residual_pred = self.policy.forward_residual_from_m(m)
            
            train_residual_loss = torch.mean((action_residual_pred - action_residuals) ** 2)
            train_residual_loss.backward()
            
            self.m_embedding_optimizer.step()
            self.residual_optimizer.step()            
            
            self.policy_optimizer.zero_grad()            
            self.h_embedding_optimizer.zero_grad()   
            
            m, h = self.policy.forward_embedding(obs)
            
            # we follow the original implementation that stop-gradient layer on m ; 
            # see `forward_policy_from_embedding` method for detail. (m.detach() in input)
            train_neg_likelihood = -self.policy.log_prob_policy_from_m_h(m, h, actions).mean()
            train_neg_likelihood.backward()
            
            self.policy_optimizer.step()
            self.h_embedding_optimizer.step()
            
            if (num+1) % eval_freq == 0:                
                valid_m, valid_h = self.policy.forward_embedding(obs_valid)                
                valid_action_residuals = actions_valid - prev_actions_valid
                valid_action_residual_pred = self.policy.forward_residual_from_m(valid_m)
            
                valid_policy_neg_likelihood = -self.policy.log_prob_policy_from_m_h(valid_m, valid_h, actions_valid).mean()
                valid_residual_loss = torch.mean((valid_action_residual_pred - valid_action_residuals) ** 2)                
                
                valid_loss = valid_policy_neg_likelihood + valid_residual_loss
                
                policy_action_valid = self.policy(obs_valid).sample()                
                
                train_mh = torch.cat([m,h], dim=-1)
                valid_mh = torch.cat([valid_m, valid_h], dim=-1)
                
                hscic_estimate = estimate_hscic(X=train_mh, Y=prev_actions, Z=actions, ridge_lambda=1e-5)
                valid_hscic_estimate = estimate_hscic(X=valid_mh, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-5)
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-5)                
                train_hscic_m_a_given_aprev = estimate_hscic(X=m, Y=actions, Z=prev_actions, ridge_lambda=1e-5)
                valid_hscic_m_a_given_aprev = estimate_hscic(X=valid_m, Y=actions_valid, Z=prev_actions_valid, ridge_lambda=1e-5)
                
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                train_loss = train_neg_likelihood + train_residual_loss
                
                print(f'** iter{num+1}: train_loss={train_loss.item()}, nll={train_neg_likelihood}, residual_loss={train_residual_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std}')
                print(f'                valid_loss={valid_loss.item()}, valid_nll={valid_policy_neg_likelihood}, valid_residual_loss={valid_residual_loss}')
                
                print(f'** HSCIC(mh, a_prev | a_current) : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                print(f'** HSCIC(m, a_current | a_prev) : (train){train_hscic_m_a_given_aprev:.6f} (valid){valid_hscic_m_a_given_aprev:.6f}  ')
                
                if self.wandb:
                    self.wandb.log({
                                    'train_total_loss':                  train_loss.item(),
                                    'valid_total_loss':                  valid_loss.item(),
                                    'train_neg_likelihood':              train_neg_likelihood.item(),
                                    'valid_neg_likelihood':              valid_policy_neg_likelihood.item(),
                                    'train_mean_hscic(rep,prev|target)': hscic_estimate,
                                    'valid_mean_hscic(rep,prev|target)': valid_hscic_estimate,
                                    'valid_mean_hscic(act,prev|target)': valid_hscic_estimate_action,
                                    'train_residual_loss':               train_residual_loss,
                                    'valid_residual_loss':               valid_residual_loss,
                                    'train_mean_hscic(m,target|prev)':   train_hscic_m_a_given_aprev,
                                    'valid_mean_hscic(m,target|prev)':   valid_hscic_m_a_given_aprev,
                                    'eval_episode_return':               eval_ret_mean
                                    }, step=num+1)

                if eval_ret_mean > max_score:
                    print(f'** max score ')
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
                # obs = obs[:true_obs_dim]
                if self.standardize:
                    obs = (obs - self.obs_mean[0]) / self.obs_std[0]
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()[0]
                next_obs, rew, done, env_info = self.env.step(action)
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
    