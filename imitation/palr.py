import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

                    
class BCwithHSCIC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, reg_coef=0.01, action_history_len=2, ridge_lambda=1e-3, standardize=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCwithHSCIC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device

        self.action_dim = action_dim
        self.reg_coef = reg_coef
            
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=mi_lr)
        
        self.num_eval_iteration = 50
        self.envname = envname
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize
        self.ridge_lambda = ridge_lambda
        self.standardize = standardize

        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std
        

    def estimate_hscic(self, X, Y, Z, ridge_lambda=1e-2, use_median=False, normalize_kernel=False, sigma2=None):
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
        
        # W = (Kz + ridge_lambda  * torch.eye(Kz.shape[0])).inverse()
        # term1 = Kz.T @ W @ (Kx * Ky) @ W.T @ Kz
        # term2 = -2 * Kz.T @ W ( Kx@W.T@Kz * Ky@W.T@Kz )
        # term3 = (Kz.T @ W @ Kx @ W.T @ Kz) * (Kz.T @ W @ Ky @ W.T @ Kz)

        return (term_1 - 2 * term_2 + term_3) / Kz.shape[0]
                    

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_train=20000, num_valid=2000, regularize_embedding=True):
        
        min_loss = 100000.
        max_score = -100000.
        hscic_scaler = None        
        
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
                if regularize_embedding:
                    policy_embedding = self.policy.forward_embedding(obs)
                    if self.standardize:
                        Y_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                        Z_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                        Y_std = Y_std.to(torch.float32)
                        Z_std = Z_std.to(torch.float32)
                    else:
                        Y_std = prev_expert_action
                        Z_std = actions
                    
                    hscic_estimate = self.estimate_hscic(X=policy_embedding, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                else:
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
                    
                    hscic_estimate = self.estimate_hscic(X=p_std, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                    # hscic_estimate = self.estimate_hscic(X=policy_action, Y=prev_expert_action, Z=actions, ridge_lambda=self.ridge_lambda)
            else:
                hscic_estimate = 0.
            
            train_loss = neg_likelihood + self.reg_coef * hscic_estimate 

            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num+1) % iter_per_valid == 0:
                policy_action = self.policy(obs).sample()
                policy_action_valid = self.policy(obs_valid).sample()
                
                if regularize_embedding:
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
                
                    hscic_estimate = self.estimate_hscic(X=policy_embedding, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                    
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
                    valid_hscic_estimate = self.estimate_hscic(X=policy_embedding_valid, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                    
                else:
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
                        
                    if self.standardize:
                        Y_std_valid = (prev_expert_action_valid - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                        Z_std_valid = (actions_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]                        
                        p_std_valid = (policy_action_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                        Y_std_valid = Y_std_valid.to(torch.float32)
                        Z_std_valid = Z_std_valid.to(torch.float32)
                        p_std_valid = p_std_valid.to(torch.float32)
                    else:
                        Y_std_valid = prev_expert_action_valid
                        Z_std_valid = actions_valid
                        p_std_valid = policy_action_valid                    
                    
                    hscic_estimate = self.estimate_hscic(X=p_std, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                    valid_hscic_estimate = self.estimate_hscic(X=p_std_valid, Y=Y_std_valid, Z=Z_std_valid, ridge_lambda=self.ridge_lambda)
                
                valid_hscic_estimate_action = self.estimate_hscic(X=policy_action_valid, Y=prev_expert_action_valid, Z=actions_valid, ridge_lambda=self.ridge_lambda)

                if hscic_scaler == None:
                    hscic_scaler = self.estimate_hscic(X=Y_std, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)

                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()
                valid_loss = valid_neg_likelihood + self.reg_coef * valid_hscic_estimate

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={train_loss.item():.2f} ({obs.shape[0]}), val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                print(f'         : (train_scaled){hscic_estimate/hscic_scaler:.6f} (valid_scaled){valid_hscic_estimate/hscic_scaler:.6f} (scaler){hscic_scaler:.6f}')
                if self.wandb:
                    self.wandb.log({'train_total_loss': train_loss.item(), 
                                    'valid_total_loss': valid_loss.item(),
                                    'train_neg_likelihood': neg_likelihood.item(),
                                    'valid_neg_likelihood': valid_neg_likelihood.item(),
                                    'train_mean_hscic_estimate': hscic_estimate,
                                    'valid_mean_hscic_estimate': valid_hscic_estimate,
                                    'valid_mean_hscic(action)': valid_hscic_estimate_action,
                                    'train_scaled_hscic': hscic_estimate/hscic_scaler,
                                    'valid_scaled_hscic': valid_hscic_estimate/hscic_scaler,
                                    # 'MINE_loss_train': mine_loss.item(), 
                                    'eval_episode_return': eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** max score record! ')
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
                    obs = (obs - self.obs_mean) / self.obs_std
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()[0]
                # if self.standardize:
                    # action = action * self.act_std[0, -self.action_dim:] + self.act_mean[0, -self.action_dim:]

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
    
