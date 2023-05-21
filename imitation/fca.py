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

from mine.mine import MINE_DV

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
                    

class FCA(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, adversarial_loss_coef=1., info_bottleneck_loss_coef=0.001):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(FCA, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
    
        self.embedding_optimizer = optim.Adam(policy.embedding_params, lr=lr)
        self.disc_optimizer = optim.Adam(policy.disc_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        
        self.device = device
        self.num_eval_iteration = 50
        self.adversarial_loss_coef = adversarial_loss_coef
        self.info_bottleneck_loss_coef = info_bottleneck_loss_coef
        self.envname = envname
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, discriminator_steps=5):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            actions = batch['actions'][:, :self.action_dim]
            prev_actions = batch['actions'][:, self.action_dim:]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer
            h = self.policy.forward_embedding(obs)

            for _ in range(discriminator_steps):
                self.disc_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                self.embedding_optimizer.zero_grad()

                # disc_log_prob = self.policy.log_prob_discriminator_from_embedding(h.detach(), actions, prev_actions)
                # disc_loss = -disc_log_prob.mean()
                
                predicted_prev_actions = self.policy.predict_prev_action_from_embedding_current_a(h, actions)            
                disc_loss = torch.mean((predicted_prev_actions - prev_actions) ** 2)

                disc_loss.backward(retain_graph=True) #retain_graph=True)       # backprop embedding
                self.disc_optimizer.step()
                # self.embedding_optimizer.step()

            self.disc_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()

            # h = self.policy.forward_embedding(obs)
            predicted_prev_actions = self.policy.predict_prev_action_from_embedding_current_a(h, actions)
            adversarial_loss = -torch.mean((predicted_prev_actions - prev_actions) ** 2)
            # adversarial_loss = self.policy.log_prob_discriminator_from_embedding(h, actions, prev_actions).mean()
            # policy_log_prob = self.policy.log_prob_policy_from_embedding(h, actions)
            # policy_neg_likelihood = -policy_log_prob.mean()
            predicted_actions = self.policy.predict_action_from_embedding(h)
            policy_neg_likelihood = torch.mean((predicted_actions - actions) ** 2)
            # policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()

            info_bottleneck_loss = 0.5 * (h ** 2).sum() 

            policy_loss = policy_neg_likelihood + \
                self.adversarial_loss_coef * adversarial_loss + \
                self.info_bottleneck_loss_coef * info_bottleneck_loss
            
            policy_loss.backward(retain_graph=True)          #retain_graph=True)     # backprop embedding
            self.policy_optimizer.step()
            self.embedding_optimizer.step()
            
            # embedding_loss.backward(retain_graph=True)     # backprop embedding
            

            if (num+1) % iter_per_valid == 0:

                # batch_valid = self.replay_buffer_valid.random_batch(batch_size)
                # obs_valid = batch_valid['observations']
                # actions = batch_valid['actions'][:, :self.action_dim]
                # prev_actions = batch_valid['actions'][:, self.action_dim:]
                # actions_valid = batch_valid['actions']
                        
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                                         
                # valid_loss = -self.policy.log_prob(obs_valid, actions_valid).mean()
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                # print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: disc_loss={disc_loss.item()}, policy_loss={policy_loss.item()}, eval_ret={eval_ret_mean}+-{eval_ret_std} ',)
                
                if self.wandb:
                    self.wandb.log({'disc_loss': disc_loss.item(), 
                                    'policy_loss': policy_loss.item(), 
                                    'info_bottleneck_loss': info_bottleneck_loss.item(),
                                    'bc_loss': policy_neg_likelihood.item(),
                                    # 'loss_valid': valid_loss,
                                    'eval_episode_return': eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** max episode score! ')
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
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()
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
                    

class FCA_Entropy(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, mi_lr=1e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, embedding_dim=1, stacksize=1, action_history_len=1,
                 reg_coef=1e-5, info_bottleneck_loss_coef=0.001, standardize=False
                 ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(FCA_Entropy, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
                
        self.device = device

        self.entropy_input_size = embedding_dim + action_dim
        self.entropy_hidden_size = 300
        self.entropy_net = nn.Sequential(
            nn.Linear(self.entropy_input_size, self.entropy_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.entropy_hidden_size, action_dim * (action_history_len-1), device=self.device)
        )

        self.embedding_optimizer = optim.Adam(policy.embedding_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        self.entropy_optimizer = optim.Adam(self.entropy_net.parameters(), lr=mi_lr)

        self.num_eval_iteration = 50
        self.entropy_coef = reg_coef 
        self.info_bottleneck_loss_coef = info_bottleneck_loss_coef
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
        

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_valid=2000, mine_steps=1):
        
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

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer
            h = self.policy.forward_embedding(obs)
            expert_action_and_h = torch.cat([actions, h], dim=-1)
            
            self.policy_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            self.entropy_optimizer.zero_grad()

            # h = self.policy.forward_embedding(obs)
            if self.entropy_coef > 1e-10:
                policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()

                prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
                pred_prev_actions = self.entropy_net(expert_action_and_h)               
                entropy_loss = torch.mean((pred_prev_actions - prev_actions) ** 2)
                # mi_estimate = self.mine.get_mi_bound(prev_actions, expert_action_and_h, update_ema=False)

                policy_loss = policy_neg_likelihood \
                    - self.entropy_coef * entropy_loss \
                    + self.info_bottleneck_loss_coef * info_bottleneck_loss
                
                policy_loss.backward(retain_graph=True)          # backprop embedding
                self.policy_optimizer.step()
                self.embedding_optimizer.step()

                for _ in range(mine_steps):
                    batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
                    obs = batch['observations']
                    actions = batch['actions'][:, -self.action_dim:]
                    prev_actions = batch['actions'][:, :-self.action_dim]
                    
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)                    
                    actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                    h = self.policy.forward_embedding(obs)
                    expert_action_and_h = torch.cat([actions, h], dim=-1)                    

                    prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
                    # pred_prev_actions = self.entropy_net(expert_action_and_h.detach())
                    pred_prev_actions = self.entropy_net(expert_action_and_h.detach())

                    entropy_loss = torch.mean((pred_prev_actions - prev_actions) ** 2)
                    
                    self.entropy_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.entropy_optimizer.step()

            else:
                policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                
                policy_loss = policy_neg_likelihood + self.info_bottleneck_loss_coef * info_bottleneck_loss
                # mine_loss = -self.mine.get_mi_bound(prev_actions, expert_action_and_h.detach(), update_ema=True)
                
                policy_loss.backward(retain_graph=True)          # backprop embedding
                self.policy_optimizer.step()
                self.embedding_optimizer.step()
                
            # embedding_loss.backward(retain_graph=True)     # backprop embedding

            if (num+1) % iter_per_valid == 0:

                # batch_valid = self.replay_buffer_valid.random_batch(batch_size, standardize=self.standardize)
                # obs_valid = batch_valid['observations']
                # actions_valid = batch_valid['actions'][:, -self.action_dim:]
                
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                
                # prev_actions = batch_valid['actions'][:, self.action_dim:]
                
                # actions_valid = batch_valid['actions']
                        
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                h = self.policy.forward_embedding(obs_valid)
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                
                if self.entropy_coef > 1e-10:
                    expert_action_and_h = torch.cat([actions_valid, h], dim=-1)  
                    pred_prev_actions = self.entropy_net(expert_action_and_h)     
                    
                    prev_actions_valid = batch_valid['actions'][:, :-self.action_dim]
                    prev_actions_valid = torch.tensor(prev_actions_valid, dtype=torch.float32, device=self.device)
                              
                    entropy_loss = torch.mean((pred_prev_actions - prev_actions_valid) ** 2)
                else:
                    entropy_loss = 0.
                    
                valid_policy_neg_likelihood = - self.policy.log_prob(obs_valid, actions_valid).mean()
                                         
                valid_loss = valid_policy_neg_likelihood \
                             - self.entropy_coef * entropy_loss \
                             + self.info_bottleneck_loss_coef * info_bottleneck_loss
                
                policy_action_valid = self.policy(obs_valid).sample()                
                h_train = self.policy.forward_embedding(obs)
                
                hscic_estimate = estimate_hscic(X=h_train, Y=prev_actions, Z=actions, ridge_lambda=1e-2)
                valid_hscic_estimate = estimate_hscic(X=h, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                # print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: entropy_loss={entropy_loss}, policy_loss={policy_loss.item()}, eval_ret={eval_ret_mean}+-{eval_ret_std} ')
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({
                                    'train_total_loss':     policy_loss.item(), 
                                    'valid_total_loss':     valid_loss.item(),
                                    'train_neg_likelihood': policy_neg_likelihood.item(),            
                                    'train_mean_hscic_estimate': hscic_estimate,                        
                                    'valid_neg_likelihood': valid_policy_neg_likelihood.item(),
                                    'valid_mean_hscic_estimate': valid_hscic_estimate,
                                    'valid_mean_hscic(action)':  valid_hscic_estimate_action,    
                                    'valid_entropy_loss':   entropy_loss, 
                                    'valid_IB_loss':        info_bottleneck_loss.item(),
                                    # 'bc_loss':              policy_neg_likelihood.item(),
                                    
                                    'eval_episode_return':  eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** min valid loss ')
                    min_loss = valid_loss
                    # policy.state_dict()
                    # max_score = eval_ret_mean
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
                action = self.policy(obs).mean.cpu().detach().numpy()
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
    

class FCA_MINE(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, mi_lr=1e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, embedding_dim=1, stacksize=1, action_history_len=1,
                 mine_loss_coef=1e-5, info_bottleneck_loss_coef=0.001, standardize=False
                 ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(FCA_MINE, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid

        self.mine = MINE_DV(action_dim * (action_history_len-1), action_dim + embedding_dim, device=device)

        self.embedding_optimizer = optim.Adam(policy.embedding_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=mi_lr)
        
        self.device = device
        self.num_eval_iteration = 50
        self.mine_loss_coef = mine_loss_coef 
        self.info_bottleneck_loss_coef = info_bottleneck_loss_coef
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
        

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_valid=2000, mine_steps=1):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
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

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer
            h = self.policy.forward_embedding(obs)
            expert_action_and_h = torch.cat([actions, h], dim=-1)
            
            self.policy_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            self.mine_optimizer.zero_grad()

            # h = self.policy.forward_embedding(obs)
            if self.mine_loss_coef > 1e-10:
                policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                mi_estimate = self.mine.get_mi_bound(prev_actions, expert_action_and_h, update_ema=False)

                policy_loss = policy_neg_likelihood + \
                    self.mine_loss_coef * mi_estimate + \
                    self.info_bottleneck_loss_coef * info_bottleneck_loss
                
                policy_loss.backward(retain_graph=True)          # backprop embedding
                self.policy_optimizer.step()
                self.embedding_optimizer.step()

                for _ in range(mine_steps):
                    batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)

                    obs = batch['observations']
                    actions = batch['actions'][:, -self.action_dim:]
                    prev_actions = batch['actions'][:, :-self.action_dim]
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

                    h = self.policy.forward_embedding(obs)                    
                    
                    actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                    prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
                    expert_action_and_h = torch.cat([actions, h], dim=-1)                    

                    mine_loss = -self.mine.get_mi_bound(prev_actions, expert_action_and_h.detach(), update_ema=True)

                    self.mine_optimizer.zero_grad()
                    mine_loss.backward()
                    self.mine_optimizer.step()

            else:
                policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                
                policy_loss = policy_neg_likelihood + self.info_bottleneck_loss_coef * info_bottleneck_loss
                # mine_loss = -self.mine.get_mi_bound(prev_actions, expert_action_and_h.detach(), update_ema=True)
                
                policy_loss.backward(retain_graph=True)          # backprop embedding
                self.policy_optimizer.step()
                self.embedding_optimizer.step()
                
            # embedding_loss.backward(retain_graph=True)     # backprop embedding

            if (num+1) % iter_per_valid == 0:

                # batch_valid = self.replay_buffer_valid.random_batch(batch_size, standardize=self.standardize)
                # obs_valid = batch_valid['observations']
                # actions_valid = batch_valid['actions'][:, -self.action_dim:]
                
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)                
                
                h = self.policy.forward_embedding(obs_valid)
                info_bottleneck_loss = 0.5 * (h ** 2).sum()
                
                if self.mine_loss_coef > 0:
                    expert_action_and_h = torch.cat([actions_valid, h], dim=-1)                                               
                    mi_estimate = self.mine.get_mi_bound(prev_actions_valid, expert_action_and_h, update_ema=False)
                else:
                    mi_estimate = 0.
                    
                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()

                valid_loss = valid_neg_likelihood + \
                    self.mine_loss_coef * mi_estimate + \
                    self.info_bottleneck_loss_coef * info_bottleneck_loss
                # actions_valid = batch_valid['actions']
                
                policy_action_valid = self.policy(obs_valid).sample()                
                h_train = self.policy.forward_embedding(obs)
                
                hscic_estimate = estimate_hscic(X=h_train, Y=prev_actions, Z=actions, ridge_lambda=1e-2)
                valid_hscic_estimate = estimate_hscic(X=h, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                valid_hscic_estimate_action = estimate_hscic(X=policy_action_valid, Y=prev_actions_valid, Z=actions_valid, ridge_lambda=1e-2)
                        
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                # print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: mine_loss={mi_estimate}, policy_loss={policy_loss.item()}, eval_ret={eval_ret_mean}+-{eval_ret_std} ')
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({
                                    'train_total_loss':          policy_loss.cpu().item(), 
                                    'valid_total_loss':          valid_loss.cpu().item(),
                                    'train_neg_likelihood':      policy_neg_likelihood.cpu().item(),
                                    'train_mean_hscic_estimate': hscic_estimate.cpu().item(),
                                    'valid_neg_likelihood':      valid_neg_likelihood.cpu().item(),                                     
                                    'valid_mean_hscic_estimate': valid_hscic_estimate.cpu().item(),
                                    'valid_mean_hscic(action)':  valid_hscic_estimate_action.cpu().item(),                                    
                                    'valid_mine_loss':           -mi_estimate.cpu().item(), 
                                    'valid_IB_loss':             info_bottleneck_loss.cpu().item(),                                    
                                    'eval_episode_return':       eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** min loss! ')                    
                    min_loss = valid_loss.item()
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
                action = self.policy(obs).mean.cpu().detach().numpy()
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
                    

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', num_recurrent_layers=1):

        super().__init__()
        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            device=device
        )
        # self.action_decoder = nn.Linear(hidden_size, action_dim, device=device)
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers
        self.hidden_state = None
        self.device = device
    
    def get_init_hidden_state(self, batch_size):
        self.hidden_state = (torch.zeros(self.num_recurrent_layers, batch_size, self.hidden_size).to(self.device),
                             torch.zeros(self.num_recurrent_layers, batch_size, self.hidden_size).to(self.device))


    def forward(self, input):
        # input shape : (seq_len, batch_size, action_dim)
        batch_size = input.shape[1]
        if self.hidden_state is None or batch_size != self.hidden_state[0].shape[1]:
            self.get_init_hidden_state(batch_size)

        _, self.hidden_state = self.lstm_encoder(input, self.hidden_state)

        # hidden_out = F.Relu(self.hidden_state)
        # hidden_out = self.action_decoder(hidden_out)
        # action_out = F.tanh(hidden_out)

        return self.hidden_state  # (0: h_t,  1: c_t)


class FCA_MINE_LSTM(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, mi_lr=1e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, embedding_dim=1, stacksize=1, action_history_len=1,
                 mine_loss_coef=1e-5, info_bottleneck_loss_coef=0.001,
                 lstm_hidden_size=32, seq_len=10,
                 ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(FCA_MINE_LSTM, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_encoder = LSTMEncoder(
            input_size=action_dim, hidden_size=lstm_hidden_size, device=device
        )
        self.mine = MINE_DV(lstm_hidden_size, action_dim+embedding_dim, device=device)

        self.embedding_optimizer = optim.Adam(policy.embedding_params, lr=lr)
        self.policy_optimizer = optim.Adam(policy.policy_params, lr=lr)
        self.mine_optimizer = optim.Adam([{'params' : self.mine.parameters()},
                                          {'params' : self.lstm_encoder.lstm_encoder.parameters()}],
                                         lr=mi_lr)
        
        self.device = device
        self.num_eval_iteration = 50
        self.mine_loss_coef = mine_loss_coef 
        self.info_bottleneck_loss_coef = info_bottleneck_loss_coef
        self.envname = envname
        self.seq_len = seq_len
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.stacksize = stacksize

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=512, mine_steps=1):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
        for num in range(0, int(total_iteration)):

            batch = self.replay_buffer.random_batch(batch_size)
            obs = batch['observations']
            actions = batch['actions'][:, :self.action_dim]
            prev_actions = batch['actions'][:, self.action_dim:]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
            prev_actions = prev_actions.view(self.seq_len, batch_size, -1)

            self.lstm_encoder.get_init_hidden_state(batch_size=batch_size)
            encoding, _ = self.lstm_encoder(prev_actions)
            h = self.policy.forward_embedding(obs)
            expert_action_and_h = torch.cat([actions, h], dim=-1)
            
            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer
            
            self.policy_optimizer.zero_grad()
            self.embedding_optimizer.zero_grad()
            self.mine_optimizer.zero_grad()

            # h = self.policy.forward_embedding(obs)            
            policy_neg_likelihood = -self.policy.log_prob_policy_from_embedding(h, actions).mean()
            info_bottleneck_loss = 0.5 * (h ** 2).sum()
            mi_estimate = self.mine.get_mi_bound(encoding[0], expert_action_and_h, update_ema=False)

            policy_loss = policy_neg_likelihood + \
                self.mine_loss_coef * mi_estimate + \
                self.info_bottleneck_loss_coef * info_bottleneck_loss
            
            policy_loss.backward(retain_graph=True)          # backprop embedding
            self.policy_optimizer.step()
            self.embedding_optimizer.step()

            for ms in range(mine_steps):
                batch = self.replay_buffer.random_batch(batch_size)
                obs = batch['observations']
                actions = batch['actions'][:, :self.action_dim]
                prev_actions = batch['actions'][:, self.action_dim:]
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                prev_actions = torch.tensor(prev_actions, dtype=torch.float32, device=self.device)
                prev_actions = prev_actions.view(self.seq_len, batch_size, -1)

                h = self.policy.forward_embedding(obs)
                expert_action_and_h = torch.cat([actions, h], dim=-1)

                self.lstm_encoder.get_init_hidden_state(batch_size=batch_size)
                encoding, _ = self.lstm_encoder(prev_actions)
                mine_loss = -self.mine.get_mi_bound(encoding[0], expert_action_and_h.detach(), update_ema=True)
                
                self.mine_optimizer.zero_grad()
                mine_loss.backward()
                self.mine_optimizer.step()

            # embedding_loss.backward(retain_graph=True)     # backprop embedding

            if (num+1) % iter_per_valid == 0:

                # batch_valid = self.replay_buffer_valid.random_batch(batch_size)
                # obs_valid = batch_valid['observations']
                # actions = batch_valid['actions'][:, :self.action_dim]
                # prev_actions = batch_valid['actions'][:, self.action_dim:]
                # actions_valid = batch_valid['actions']
                        
                # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                                         
                # valid_loss = -self.policy.log_prob(obs_valid, actions_valid).mean()
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                # print(f'** iter{num+1}: train_loss={loss} ({obs.shape[0]}), val_loss={valid_loss}, eval_ret={eval_ret_mean}+-{eval_ret_std} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: mine_loss={mine_loss.item()}, policy_loss={policy_loss.item()}, eval_ret={eval_ret_mean}+-{eval_ret_std} ')
                
                if self.wandb:
                    self.wandb.log({'mine_loss':            mine_loss.item(), 
                                    'policy_loss':          policy_loss.item(), 
                                    'info_bottleneck_loss': info_bottleneck_loss.item(),
                                    'bc_loss':              policy_neg_likelihood.item(),
                                    # 'loss_valid': valid_loss,
                                    'eval_episode_return':  eval_ret_mean
                                    }, step=num)

                # if valid_loss < min_loss:
                if eval_ret_mean > max_score:
                    print(f'** max episode score! ')
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
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()
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
                    