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
        

def compute_pdist_sq(x, y=None):
    """compute the squared paired distance between x and y."""
    if y is not None:
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
    a = x.view(x.shape[0], -1)
    aTa = torch.mm(a, a.T)
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)

def gaussian_kernel(X, sigma2=None, Y=None, normalized=False, **ignored):
    if normalized:
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)
        if Y is not None:
            Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)
    Dxx = compute_pdist_sq(X, Y)
    if sigma2 is None:
        sigma2 = Dxx.median()
    Kx = torch.exp(-Dxx / sigma2)
    return Kx

def estimate_hscic(X, Y, Z, ridge_lambda=1e-2, use_median=False, normalize_kernel=False, sigma2=None):
    '''X ind. Y | Z '''
    # (1) action regularization version : X = imitator action
    # (2) regularized representation version : X = varphi(Obs)

    if sigma2 is None:
        if use_median:
            sigma2_ = None
        else:
            sigma2_ = 1.
    else:
        sigma2_ = sigma2

    Kx = gaussian_kernel(X, sigma2=sigma2_, normalized=normalize_kernel)
    Ky = gaussian_kernel(Y, sigma2=sigma2_, normalized=normalize_kernel)
    Kz = gaussian_kernel(Z, sigma2=sigma2_, normalized=normalize_kernel)

    # https://arxiv.org/pdf/2207.09768.pdf
    WtKzz = torch.linalg.solve(Kz + ridge_lambda  * torch.eye(Kz.shape[0]).to(Kz.device), Kz) # * Kz.shape[0] for ridge_lambda
    
    term_1 = (WtKzz * ((Kx * Ky) @ WtKzz)).sum()    # tr(WtKzz.T @ (Kx * Ky) @ WtKzz)
    WkKxWk = WtKzz * (Kx @ WtKzz)
    KyWk = Ky @ WtKzz
    term_2 = (WkKxWk * KyWk).sum()
    
    term_3 = (WkKxWk.sum(dim=0) * (WtKzz * KyWk).sum(dim=0)).sum()

    return (term_1 - 2 * term_2 + term_3) / Kz.shape[0]

class RAP(nn.Module):
    # Implementation of Residual Action Prediction (ECCV 2022)
    # - https://arxiv.org/pdf/2207.09705.pdf
    # - The author code is not released
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
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
        # self.entropy_coef = reg_coef 
        # self.info_bottleneck_loss_coef = info_bottleneck_loss_coef
        self.envname = envname
        
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
        

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_valid=2000):
        
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
            
            m, _ = self.policy.forward_embedding(obs)            
            action_residuals = actions - prev_actions
            action_residual_pred = self.policy.forward_residual_from_m(m)
            
            residual_loss = torch.mean((action_residual_pred - action_residuals) ** 2)
            residual_loss.backward()
            
            self.m_embedding_optimizer.step()
            self.residual_optimizer.step()            
            
            self.policy_optimizer.zero_grad()            
            self.h_embedding_optimizer.zero_grad()   
            
            m, h = self.policy.forward_embedding(obs)
            policy_neg_likelihood = -self.policy.log_prob_policy_from_m_h(m, h, actions).mean()                                        
            # train_loss.backward()          # backprop embedding
            policy_neg_likelihood.backward()
            
            self.policy_optimizer.step()            
            self.h_embedding_optimizer.step()         
        
            # residual_loss = -self.policy.log_prob_residual_from_m(m, action_residuals).mean()            
            # train_loss = policy_neg_likelihood + residual_loss   
            
            if (num+1) % iter_per_valid == 0:                
                valid_m, valid_h = self.policy.forward_embedding(obs_valid)                
                valid_action_residuals = actions_valid - prev_actions_valid
                valid_action_residual_pred = self.policy.forward_residual_from_m(valid_m)
            
                valid_policy_neg_likelihood = -self.policy.log_prob_policy_from_m_h(valid_m, valid_h, actions_valid).mean()
                valid_residual_loss = torch.mean((valid_action_residual_pred - valid_action_residuals) ** 2)
                # valid_residual_loss = -self.policy.log_prob_residual_from_m(valid_m, valid_action_residuals).mean() 
                
                
                valid_loss = valid_policy_neg_likelihood + valid_residual_loss
                
                policy_action_valid = self.policy(obs_valid).sample()                
                # mh = self.policy.forward_embedding(obs)
                
                train_mh = torch.cat([m,h], dim=-1)
                valid_mh = torch.cat([valid_m, valid_h], dim=-1)
                
                hscic_estimate = estimate_hscic(X=train_mh, Y=prev_actions, Z=actions, ridge_lambda=1e-2)
                valid_hscic_estimate = estimate_hscic(X=valid_mh, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                
                train_hscic_m_a_given_aprev = estimate_hscic(X=m, Y=actions, Z=prev_actions, ridge_lambda=1e-2)
                valid_hscic_m_a_given_aprev = estimate_hscic(X=valid_m, Y=actions_valid, Z=prev_actions_valid, ridge_lambda=1e-2)
                
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                train_loss = policy_neg_likelihood + residual_loss
                
                # print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: train_loss={train_loss.item()}, nll={policy_neg_likelihood}, residual_loss={residual_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std}')
                print(f'                valid_loss={valid_loss.item()}, valid_nll={valid_policy_neg_likelihood}, valid_residual_loss={valid_residual_loss}')
                
                print(f'** HSCIC(mh, a_prev | a_current) : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                print(f'** HSCIC(m, a_current | a_prev) : (train){train_hscic_m_a_given_aprev:.6f} (valid){valid_hscic_m_a_given_aprev:.6f}  ')
                
                if self.wandb:
                    self.wandb.log({
                                    'train_total_loss':             train_loss.item(), 
                                    'valid_total_loss':             valid_loss.item(),
                                    'train_neg_likelihood':         policy_neg_likelihood.item(),            
                                    'train_mean_hscic_estimate':    hscic_estimate,                        
                                    'valid_neg_likelihood':         valid_policy_neg_likelihood.item(),
                                    'valid_mean_hscic_estimate':    valid_hscic_estimate,
                                    'valid_mean_hscic(action)':     valid_hscic_estimate_action,    
                                    'train_residual_loss':          residual_loss,
                                    'valid_residual_loss':          valid_residual_loss,
                                    'train_hscic_residual':         train_hscic_m_a_given_aprev,
                                    'valid_hscic_residual':         valid_hscic_m_a_given_aprev,
                                    # 'bc_loss':                    policy_neg_likelihood.item(),                                    
                                    'eval_episode_return':          eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** max score ')
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
    