import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

from imitation.bc import copy_nn_module
from core.hscic import estimate_hscic

class FCA(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True,
                 embedding_dim=1, entropy_hidden_size=300, entropy_lr=1e-4, reg_coef=1e-5, info_bottleneck_loss_coef=0.001, 
                 ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(FCA, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid                
        self.device = device
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.stacksize = stacksize        

        # Additional Network for Conditional Entropy (FCA)
        self.entropy_input_size = embedding_dim + action_dim
        self.entropy_hidden_size = entropy_hidden_size
        self.entropy_net = nn.Sequential(
            nn.Linear(self.entropy_input_size, self.entropy_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.entropy_hidden_size, action_dim, device=self.device)
        )
        
        # FCA Hyperparameters
        self.entropy_coef = reg_coef 
        self.info_bottleneck_loss_coef = info_bottleneck_loss_coef        
        
        self.embedding_optimizer = optim.Adam(policy.embedding_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        self.entropy_optimizer = optim.Adam(self.entropy_net.parameters(), lr=entropy_lr)

        self.num_eval_iteration = 50
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

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

    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024, num_valid=2000, inner_steps=1):
        
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

            # conditional entropy input : H(a_{t-1}| a_{t}, varphi_t)
            h = self.policy.forward_embedding(obs)
            expert_action_and_h = torch.cat([actions, h], dim=-1) 
            
            self.policy_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            self.entropy_optimizer.zero_grad()

            if self.entropy_coef > 0.:
                neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()

                # prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
                pred_prev_actions = self.entropy_net(expert_action_and_h)               
                entropy_loss = torch.mean((pred_prev_actions - prev_actions) ** 2)                

                train_loss =    neg_likelihood \
                              - self.entropy_coef * entropy_loss \
                              + self.info_bottleneck_loss_coef * info_bottleneck_loss
                
                train_loss.backward()          # backprop embedding
                
                self.policy_optimizer.step()
                self.embedding_optimizer.step()

                # conditional entropy training
                for _ in range(inner_steps):
                    batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
                    
                    obs = batch['observations']
                    actions = batch['actions'][:, -self.action_dim:]
                    prev_actions = batch['actions'][:, :-self.action_dim]
                    
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)                    
                    actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                    
                    h = self.policy.forward_embedding(obs)
                    expert_action_and_h = torch.cat([actions, h], dim=-1)                    

                    prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)                    
                    pred_prev_actions = self.entropy_net(expert_action_and_h.detach())

                    entropy_loss = torch.mean((pred_prev_actions - prev_actions) ** 2)
                    
                    self.entropy_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.entropy_optimizer.step()

            else:
                neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                
                train_loss = neg_likelihood + self.info_bottleneck_loss_coef * info_bottleneck_loss                
                
                train_loss.backward()
                
                self.policy_optimizer.step()
                self.embedding_optimizer.step()    
            

            if (num+1) % eval_freq == 0:                
                h_valid = self.policy.forward_embedding(obs_valid)
                valid_info_bottleneck_loss = 0.5 * (h_valid ** 2).sum()
                
                if self.entropy_coef > 0:
                    expert_action_and_h_valid = torch.cat([actions_valid, h_valid], dim=-1)  
                    pred_prev_actions_valid = self.entropy_net(expert_action_and_h_valid)
                    
                    prev_actions_valid = batch_valid['actions'][:, :-self.action_dim]
                    prev_actions_valid = torch.tensor(prev_actions_valid, dtype=torch.float32, device=self.device)
                              
                    valid_entropy_loss = torch.mean((pred_prev_actions_valid - prev_actions_valid) ** 2)
                else:
                    valid_entropy_loss = 0.
                    
                valid_neg_likelihood = - self.policy.log_prob(obs_valid, actions_valid).mean()
                                         
                valid_loss = valid_neg_likelihood \
                             - self.entropy_coef * valid_entropy_loss \
                             + self.info_bottleneck_loss_coef * valid_info_bottleneck_loss
                
                policy_action_valid = self.policy(obs_valid).sample()                
                h_train = self.policy.forward_embedding(obs)
                
                hscic_estimate = estimate_hscic(X=h_train, Y=prev_actions, Z=actions, ridge_lambda=1e-5)
                valid_hscic_estimate = estimate_hscic(X=h_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-5)
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-5)
                
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: entropy_loss={entropy_loss}, train_loss={train_loss.item()}, eval_ret={eval_ret_mean}+-{eval_ret_std} ')
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({
                                    'train_total_loss':                  train_loss.item(), 
                                    'valid_total_loss':                  valid_loss.item(),
                                    'train_neg_likelihood':              neg_likelihood.item(),            
                                    'valid_neg_likelihood':              valid_neg_likelihood.item(),
                                    'train_mean_hscic(rep,prev|target)': hscic_estimate,
                                    'valid_mean_hscic(rep,prev|target)': valid_hscic_estimate,
                                    'valid_mean_hscic(act,prev|target)': valid_hscic_estimate_action,
                                    'valid_entropy_loss':                entropy_loss, 
                                    'valid_IB_loss':                     info_bottleneck_loss.item(),
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
    

