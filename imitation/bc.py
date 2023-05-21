import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# from mine.models.mine import Mine # https://github.com/gtegner/mine-pytorch
from mine.mine import MINE_DV
from scipy.linalg import solve as scp_solve

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


class BCwithMIReg(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, mi_lr=1e-3, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, mi_reg_coef=0.01, alternate=False, action_history_len=2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCwithMIReg, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device

        self.action_dim = action_dim
        # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
        self.mi_reg_coef = mi_reg_coef
        # self.mine_network = nn.Sequential(
        #     nn.Linear(action_dim*3, 300, device=self.device),
        #     nn.ReLU(),
        #     nn.Linear(300, 100, device=self.device),
        #     nn.ReLU(),
        #     nn.Linear(100, 1, device=self.device))

        # self.mine = Mine(
        #     T = self.mine_network ,
        #     loss = 'mine', #mine_biased, fdiv
        #     method = 'concat'
        # )
        self.mine = MINE_DV(action_dim * action_history_len, action_dim*2, device=device)
    
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=mi_lr)

        self.device = device
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

    def train(self, total_iteration=1e6, iter_per_valid=1000, 
              iter_per_mi_estimate=1e3, mi_iteration=100, batch_size=64, num_train=20000, alternate=False):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
        for num in range(0, int(total_iteration)):
            if alternate:
                batch = self.replay_buffer.random_batch(batch_size)

                obs = batch['observations']
                actions = batch['actions'][:, :self.action_dim]
                prev_expert_action = batch['actions'][:, self.action_dim:]
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)
                cur_imitator_action = self.policy(obs).rsample() # reparam
                cur_action = torch.cat([actions, cur_imitator_action], dim=-1)

                neg_likelihood = -self.policy.log_prob(obs, actions).mean()

                # NOTE:
                # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
                # policy_action = self.policy(obs).rsample()
                # cur_action = torch.cat([actions, policy_action], dim=-1)
                mi_estimate = self.mine.get_mi_bound(prev_expert_action, cur_action, update_ema=False)

                bc_loss = neg_likelihood + self.mi_reg_coef * mi_estimate

                self.policy_optimizer.zero_grad() 
                bc_loss.backward(retain_graph=True)
                self.policy_optimizer.step()

                mine_loss = -self.mine.get_mi_bound(prev_expert_action, cur_action.detach(), update_ema=True)

                self.mine_optimizer.zero_grad()
                mine_loss.backward()
                self.mine_optimizer.step()

            else:
                if (num) % iter_per_mi_estimate == 0:
                    # for _ in range(mi_iteration):
                    print('--start MINE training')
                    batch = self.replay_buffer.random_batch(num_train)
                    prev_expert_action = batch['actions'][:, -self.action_dim:]
                    cur_expert_action = batch['actions'][:, :self.action_dim]

                    prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)
                    cur_expert_action = torch.tensor(cur_expert_action, dtype=torch.float32, device=self.device)
                    
                    obs = batch['observations']
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    cur_imitator_action = self.policy(obs).sample() # not rsample

                    cur_action = torch.cat([cur_expert_action, cur_imitator_action], dim=-1)

                    self.mine.optimize(X=prev_expert_action, Y=cur_action, iters=int(mi_iteration), batch_size=256)
                    print('--end MINE training')

                batch = self.replay_buffer.random_batch(batch_size)

                obs = batch['observations']
                actions = batch['actions'][:, :self.action_dim]
                prev_expert_action = batch['actions'][:, -self.action_dim:]
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)

                # NOTE:
                # obs = obs[:,obs_idx] -> processed in replay buffer
                # actions = torch.tanh(actions) -> processed in replay buffer          
                neg_likelihood = - self.policy.log_prob(obs, actions).mean()
                
                # NOTE:
                # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
                policy_action = self.policy(obs).rsample()
                cur_action = torch.cat([actions, policy_action], dim=-1)
                mi_estimate = self.mine.mi(prev_expert_action, cur_action)

                loss = neg_likelihood + self.mi_reg_coef * mi_estimate

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (num+1) % iter_per_valid == 0:

                batch_valid = self.replay_buffer_valid.random_batch(batch_size)
                obs_valid = batch_valid['observations']
                actions_valid = batch_valid['actions'][:, :self.action_dim]
                        
                obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                                         
                valid_loss = -self.policy.log_prob(obs_valid, actions_valid).mean()
                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={bc_loss.item():.2f} ({obs.shape[0]}), mine_loss={mine_loss.item():.2f}, val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                
                if self.wandb:
                    self.wandb.log({'BC_loss_train': bc_loss.item(), 
                                    'BC_loss_valid': valid_loss.item(),
                                    'MINE_loss_train': mine_loss.item(), 
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
        
        # actions = batch['actions'][:, -self.action_dim:]
        # prev_expert_action = batch['actions'][:, :-self.action_dim]
    
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

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer          
            neg_likelihood = - self.policy.log_prob(obs, actions).mean()
            
            # NOTE:
            # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
            policy_action = self.policy(obs).rsample()
            # cur_action = torch.cat([actions, policy_action], dim=-1)

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
    

class BCwithHSIC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, reg_coef=0.01, action_history_len=2, ridge_lambda=1e-3, standardize=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCwithHSIC, self).__init__()

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
        

    def hsic_matrices(self, Kx, Ky, biased=False):
        n = Kx.shape[0]

        if biased:
            a_vec = Kx.mean(dim=0)
            b_vec = Ky.mean(dim=0)
            # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
            return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

        else:
            tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
            tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

            u = tilde_Kx * tilde_Ky
            k_row = tilde_Kx.sum(dim=1)
            l_row = tilde_Ky.sum(dim=1)
            mean_term_1 = u.sum()  # tr(KL)
            mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
            mu_x = tilde_Kx.sum()
            mu_y = tilde_Ky.sum()
            mean_term_3 = mu_x * mu_y

            # Unbiased HISC.
            mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean

    def estimate_hsic(self, X, Y, Kx_sigma2=1., Ky_sigma2=1., biased=False):
        '''X ind. Y'''
        # todo:
        #  alternative implementation for RFF
        #  biased/unbiased HSIC choice
        #  faster implementation for biased
        Kx = gaussian_kernel(X, sigma2=Kx_sigma2)
        Ky = gaussian_kernel(Y, sigma2=Ky_sigma2)
       
        return self.hsic_matrices(Kx, Ky, biased)
    
    def estimate_hscic(self, X, Y, Z, ridge_lambda=1e-2, use_median=False):
        '''X ind. Y | Z '''
        # (1) action regularization version : X = imitator action
        # (2) regularized representation version : X = varphi(Obs)

        if use_median:
            sigma2 = None
        else:
            sigma2 = 1.

        Kx = gaussian_kernel(X, sigma2=sigma2)
        Ky = gaussian_kernel(Y, sigma2=sigma2)
        Kz = gaussian_kernel(Z, sigma2=sigma2)

        # https://arxiv.org/pdf/2207.09768.pdf
        WtKzz = torch.linalg.solve(Kz + ridge_lambda  * torch.eye(Kz.shape[0]).to(Kz.device), Kz) # * Kz.shape[0] for ridge_lambda
       
        term_1 = (WtKzz * ((Kx * Ky) @ WtKzz)).sum()    # tr(WtKzz.T @ (Kx * Ky) @ WtKzz)
        WkKxWk = WtKzz * (Kx @ WtKzz)
        KyWk = Ky @ WtKzz
        term_2 = (WkKxWk * KyWk).sum()
        
        term_3 = (WkKxWk.sum(dim=0) * (WtKzz * KyWk).sum(dim=0)).sum()

        return (term_1 - 2 * term_2 + term_3) / Kz.shape[0]
                    

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_train=20000, num_valid=2000, regularize_embedding=True):
        
        min_loss = 100000.
        max_score = -100000.
        hscic_scaler = None        
        
        batch_valid = self.replay_buffer_valid.get_batch(num_valid, standardize=self.standardize)
        obs_valid = batch_valid['observations']
        
        # actions = batch['actions'][:, -self.action_dim:]
        # prev_expert_action = batch['actions'][:, :-self.action_dim]
    
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

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer          
            neg_likelihood = - self.policy.log_prob(obs, actions).mean()
            
            # NOTE:
            # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
            policy_action = self.policy(obs).rsample()
            # cur_action = torch.cat([actions, policy_action], dim=-1)

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
                    
                    hsic_estimate = self.estimate_hsic(X=policy_embedding, Y=Y_std)
                else:
                    hsic_estimate = self.estimate_hsic(X=policy_action, Y=prev_expert_action)
            else:
                hsic_estimate = 0.
            
            train_loss = neg_likelihood + self.reg_coef * hsic_estimate 

            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num+1) % iter_per_valid == 0:
                policy_action_valid = self.policy(obs_valid).sample()
                
                if regularize_embedding:
                    # Train data HSCIC (for debugging)                    
                    policy_embedding = self.policy.forward_embedding(obs)
                    if self.standardize:
                        Y_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                        Z_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                        Y_std = Y_std.to(torch.float32)
                        Z_std = Z_std.to(torch.float32)
                    else:
                        Y_std = prev_expert_action
                        Z_std = actions
                
                    hsic_estimate = self.estimate_hsic(X=policy_embedding, Y=Y_std)
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
                    
                    valid_hsic_estimate = self.estimate_hsic(X=policy_embedding_valid, Y=Y_std)
                    valid_hscic_estimate = self.estimate_hscic(X=policy_embedding_valid, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)
                else:
                    hsic_estimate = self.estimate_hsic(X=policy_action, Y=prev_expert_action)
                    valid_hsic_estimate = self.estimate_hsic(X=policy_action_valid, Y=prev_expert_action_valid)
                    
                    hscic_estimate = self.estimate_hscic(X=policy_action, Y=prev_expert_action, Z=actions, ridge_lambda=self.ridge_lambda)
                    valid_hscic_estimate = self.estimate_hscic(X=policy_action_valid, Y=prev_expert_action_valid, Z=actions_valid, ridge_lambda=self.ridge_lambda)
                
                valid_hscic_estimate_action = self.estimate_hscic(X=policy_action_valid, Y=prev_expert_action_valid, Z=actions_valid, ridge_lambda=self.ridge_lambda)

                if hscic_scaler == None:
                    hscic_scaler = self.estimate_hscic(X=Y_std, Y=Y_std, Z=Z_std, ridge_lambda=self.ridge_lambda)

                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()
                valid_loss = valid_neg_likelihood + self.reg_coef * valid_hsic_estimate

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={train_loss.item():.2f} ({obs.shape[0]}), val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'** HSCIC : (train){hscic_estimate:.6f} (valid){valid_hscic_estimate:.6f} (valid,action){valid_hscic_estimate_action:.6f}')
                print(f'         : (train_scaled){hscic_estimate/hscic_scaler:.6f} (valid_scaled){valid_hscic_estimate/hscic_scaler:.6f} (scaler){hscic_scaler:.6f}')
                print(f'** HSIC  : (train){hsic_estimate:.6f} (valid){valid_hsic_estimate:.6f}')
                
                if self.wandb:
                    self.wandb.log({'train_total_loss': train_loss.item(), 
                                    'valid_total_loss': valid_loss.item(),
                                    'train_neg_likelihood': neg_likelihood.item(),
                                    'valid_neg_likelihood': valid_neg_likelihood.item(),
                                    'train_mean_hscic_estimate': hscic_estimate,
                                    'valid_mean_hscic_estimate': valid_hscic_estimate,
                                    'valid_mean_hscic(action)': valid_hscic_estimate_action,
                                    'train_mean_hsic_estimate': hsic_estimate,
                                    'valid_mean_hsic_estimate': valid_hsic_estimate,
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
                    
class BCwithHSIC_Z_XY(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, reg_coef=0.01, action_history_len=2, standardize=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCwithHSIC_Z_XY, self).__init__()

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
        self.standardize = standardize
        self.action_standardize = False

        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device, dtype=torch.float32)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device, dtype=torch.float32)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device, dtype=torch.float32)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device, dtype=torch.float32)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std
        

    def hsic_matrices(self, Kx, Ky, biased=False):
        n = Kx.shape[0]

        if biased:
            a_vec = Kx.mean(dim=0)
            b_vec = Ky.mean(dim=0)
            # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
            return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

        else:
            tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
            tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

            u = tilde_Kx * tilde_Ky
            k_row = tilde_Kx.sum(dim=1)
            l_row = tilde_Ky.sum(dim=1)
            mean_term_1 = u.sum()  # tr(KL)
            mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
            mu_x = tilde_Kx.sum()
            mu_y = tilde_Ky.sum()
            mean_term_3 = mu_x * mu_y

            # Unbiased HISC.
            mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean

    def estimate_hsic(self, X, Y, Kx_sigma2=1., Ky_sigma2=1., biased=False):
        '''X ind. Y'''
        # todo:
        #  alternative implementation for RFF
        #  biased/unbiased HSIC choice
        #  faster implementation for biased
        Kx = gaussian_kernel(X, sigma2=Kx_sigma2)
        Ky = gaussian_kernel(Y, sigma2=Ky_sigma2)
       
        return self.hsic_matrices(Kx, Ky, biased)
    
    def estimate_hscic(self, X, Y, Z, ridge_lambda=1e-2, use_median=False):
        '''X ind. Y | Z '''
        # (1) action regularization version : X = imitator action
        # (2) regularized representation version : X = varphi(Obs)

        if use_median:
            sigma2 = None
        else:
            sigma2 = 1.

        Kx = gaussian_kernel(X, sigma2=sigma2)
        Ky = gaussian_kernel(Y, sigma2=sigma2)
        Kz = gaussian_kernel(Z, sigma2=sigma2)

        # https://arxiv.org/pdf/2207.09768.pdf
        WtKzz = torch.linalg.solve(Kz + ridge_lambda  * torch.eye(Kz.shape[0]).to(Kz.device), Kz) # * Kz.shape[0] for ridge_lambda
       
        term_1 = (WtKzz * ((Kx * Ky) @ WtKzz)).sum()    # tr(WtKzz.T @ (Kx * Ky) @ WtKzz)
        WkKxWk = WtKzz * (Kx @ WtKzz)
        KyWk = Ky @ WtKzz
        term_2 = (WkKxWk * KyWk).sum()
        
        term_3 = (WkKxWk.sum(dim=0) * (WtKzz * KyWk).sum(dim=0)).sum()

        return (term_1 - 2 * term_2 + term_3) / Kz.shape[0]


    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_train=20000, num_valid=2000, regularize_embedding=True):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)

            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            prev_expert_action = batch['actions'][:, :-self.action_dim]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer          
            neg_likelihood = - self.policy.log_prob(obs, actions).mean()
            
            # NOTE:
            # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
            policy_action = self.policy(obs).rsample()
            # cur_action = torch.cat([actions, policy_action], dim=-1)

            if self.reg_coef > 0:
                if regularize_embedding:
                    policy_embedding = self.policy.forward_embedding(obs)
                    if self.action_standardize:
                        Y_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]
                        Z_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                        # policy_action_std = (policy_action - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]
                        
                        Y_std = Y_std.to(torch.float32)
                        Z_std = Z_std.to(torch.float32)                        
                        
                    else:
                        Y_std = actions
                        Z_std = prev_expert_action
                        
                    XY = torch.cat([policy_embedding, Y_std], dim=1)                    
                    hsic_estimate = self.estimate_hsic(X=Z_std, Y=XY)
                else:
                    XY = torch.cat([policy_action, actions], dim=1)
                    hsic_estimate = self.estimate_hsic(X=prev_expert_action, Y=XY)
            else:
                hsic_estimate = 0.
            
            train_loss = neg_likelihood + self.reg_coef * hsic_estimate 

            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num+1) % iter_per_valid == 0:

                batch_valid = self.replay_buffer_valid.random_batch(num_valid, standardize=self.standardize)
                obs_valid = batch_valid['observations']
                
                # actions = batch['actions'][:, -self.action_dim:]
                # prev_expert_action = batch['actions'][:, :-self.action_dim]
            
                actions_valid = batch_valid['actions'][:, -self.action_dim:]
                prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim]
                        
                obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)
                policy_action_valid = self.policy(obs_valid).sample()

                if regularize_embedding:
                    policy_embedding_valid = self.policy.forward_embedding(obs_valid)
                    if self.action_standardize:
                        Z_std = (prev_expert_action_valid - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                        Y_std = (actions_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]
                        
                        policy_action_valid_std = (policy_action_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                        Y_std = Y_std.to(torch.float32)
                        Z_std = Z_std.to(torch.float32)
                        # policy_action_valid_std = policy_action_valid_std.to(torch.float32)
                        
                    else:
                        Z_std = prev_expert_action_valid
                        Y_std = actions_valid
                        policy_action_valid_std = policy_action_valid
                        
                    XY = torch.cat([policy_embedding_valid, Y_std], dim=1)                    
                    valid_hsic_estimate = self.estimate_hsic(X=Z_std, Y=XY)
                else:
                    XY = torch.cat([policy_action_valid, actions_valid], dim=1)
                    valid_hsic_estimate = self.estimate_hsic(X=prev_expert_action_valid, Y=XY)
                
                XY = torch.cat([policy_action_valid_std, Y_std], dim=1)
                valid_hsic_estimate_action = self.estimate_hsic(X=prev_expert_action_valid, Y=XY)                
                valid_hsic_Z_Y = self.estimate_hsic(X=Z_std, Y=Y_std)
                
                valid_hscic_estimate = self.estimate_hscic(X=policy_embedding_valid, Y=Z_std, Z=Y_std)
                valid_hscic_estimate_action = self.estimate_hscic(X=policy_action_valid_std, Y=Z_std, Z=Y_std)

                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()
                valid_loss = valid_neg_likelihood + self.reg_coef * valid_hsic_estimate

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={train_loss.item():.2f} ({obs.shape[0]}), val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'** HSIC : (train,Z:XY){hsic_estimate:.6f} (valid,Z:XY){valid_hsic_estimate:.6f} (valid,Z:Y){valid_hsic_Z_Y:.6f}(valid,action){valid_hsic_estimate_action:.6f}')
                print(f'** HSCIC : (valid,XY|Z){valid_hscic_estimate:.6f} (valid,AY|Z) {valid_hscic_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({'BC_loss_train': train_loss.item(), 
                                    'BC_loss_valid': valid_loss.item(),
                                    'train_neg_likelihood': neg_likelihood.item(),
                                    'valid_neg_likelihood': valid_neg_likelihood.item(),
                                    'train_mean_hsic_Z_XY': hsic_estimate,
                                    'valid_mean_hsic_Z_XY': valid_hsic_estimate,
                                    'valid_mean_hsic_Z_Y': valid_hsic_Z_Y,
                                    'valid_mean_hsic(action)': valid_hsic_estimate_action,
                                    'valid_mean_hscic': valid_hscic_estimate,
                                    'valid_mean_hscic(action)': valid_hscic_estimate_action,
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
                    


class BCwithCIRCE(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, replay_buffer_heldout=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, reg_coef=0.01, action_history_len=2, ridge_lambda=1e-3, standardize=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCwithCIRCE, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.replay_buffer_heldout = replay_buffer_heldout
        
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
        # self.ridge_lambda = ridge_lambda
        
        self.standardize = standardize
        self.action_standardize = False
        
        
        self.ridge_lambda = ridge_lambda
        self.sigma2 = None
        self.loo_cond_mean = True
        
        self.kx_sigma = 1.
        self.ky_sigma = None
        self.kz_sigma = 10.
        
        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std
        

    def hsic_matrices(self, Kx, Ky, biased=False):
        n = Kx.shape[0]

        if biased:
            a_vec = Kx.mean(dim=0)
            b_vec = Ky.mean(dim=0)
            # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
            return (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()

        else:
            tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
            tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

            u = tilde_Kx * tilde_Ky
            k_row = tilde_Kx.sum(dim=1)
            l_row = tilde_Ky.sum(dim=1)
            mean_term_1 = u.sum()  # tr(KL)
            mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
            mu_x = tilde_Kx.sum()
            mu_y = tilde_Ky.sum()
            mean_term_3 = mu_x * mu_y

            # Unbiased HISC.
            mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
            return mean


    def estimate_circe(self, X, Z, Z_heldout, Y, Y_heldout, W_1, W_2, biased=False, cond_cov=False, best_sigma2=None,
                       kx_sigma=0.01, ky_sigma=None, kz_sigma=0.01):
        '''X ind. Z | Y '''
        # (1) action regularization version : X = imitator action
        # (2) regularized representation version : X = varphi(Obs)
        # Z = prev. data action
        # Y = data action

        Z_all = torch.vstack((Z, Z_heldout))
        Kz_all = gaussian_kernel(Z_all, Y=Z, sigma2=kz_sigma)

        if ky_sigma is None:
            ky_sigma = best_sigma2

        Y_all = torch.vstack((Y, Y_heldout))
        Ky_all = gaussian_kernel(Y_all, Y=Y, sigma2=ky_sigma)

        Kx = gaussian_kernel(X, sigma2=kx_sigma)

        n_points = Y.shape[0]

        A = (0.5 * Ky_all[n_points:, :].T @ W_2 - Kz_all[n_points:, :].T) @ W_1 @ Ky_all[n_points:, :]
        Kres = Kz_all[:n_points, :n_points] + A + A.T
        Kres = Kres * Ky_all[:n_points, :]

        if cond_cov:
            Kx = Kx * Kres
            if biased:
                return Kx.mean()
            idx = torch.triu_indices(n_points, n_points, 1)
            return Kx[idx[0], idx[1]].mean()
            
        return self.hsic_matrices(Kx, Kres, biased)

    def leave_one_out_reg(self, K_YY, labels, reg_list):
        U, eigs = np.linalg.svd(K_YY, hermitian=True)[:2]
        regs = np.array(reg_list)
        eigs_reg_inv = 1 / (eigs[:, None] + regs[None, :]) # rank x n_regs

        KU = U * eigs[None, :]#K_YY @ U
        Ul = U.T @ labels
        preds = np.tensordot(KU, Ul[:, :, None] * eigs_reg_inv[:, None, :], axes=1) # rank x label_dim x n_regs
        # A = (U * eigs / (eigs + reg)) @ U.T
        A_ii = (U ** 2 @ (eigs[:, None] * eigs_reg_inv))  # rank x n_regs

        return np.mean(((labels[:, :, None] - preds) / (1 - A_ii)[:, None, :]) ** 2, axis=(0, 1))
    
    def leave_one_out_reg_kernels_one(self, K_YY, K_QQ, reg):
        Kinv = np.linalg.solve(K_YY + reg * np.eye(K_YY.shape[0]), K_YY).T
        diag_idx = np.arange(K_YY.shape[0])
        return ((K_QQ[diag_idx, diag_idx] + (Kinv @ K_QQ @ Kinv)[diag_idx, diag_idx] -
                2 * (Kinv @ K_QQ)[diag_idx, diag_idx]) / (1 - Kinv[diag_idx, diag_idx]) ** 2).mean()
    
    def leave_one_out_reg_kernels(self, K_YY, K_QQ, reg_list):
        loos = []
        for reg in reg_list:
            loos.append(self.leave_one_out_reg_kernels_one(K_YY, K_QQ, reg))
        U, eigs = np.linalg.svd(K_YY, hermitian=True)[:2]
        svd_tol = eigs.max() * U.shape[0] * np.finfo(U.dtype).eps
        regs = np.array(reg_list)
        return loos, regs < svd_tol, svd_tol
    
    def _leave_one_out_regressors(self, Y_heldout, reg_list, sigma2_list, Kz):
        LOO_error_sanity_check = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_error = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_tol = np.zeros(len(sigma2_list))
        for idx, sigma2 in enumerate(sigma2_list):
            print(idx, sigma2)
            # self.kernel_y_args['sigma2'] = sigma2
            loo_size = Y_heldout.shape[0] // 2
            # K_YY = eval(f'losses.{self.kernel_y}_kernel(self.Y_heldout, **self.kernel_y_args)')
            K_YY = gaussian_kernel(Y_heldout, sigma2=sigma2)
            LOO_error_sanity_check[idx] = self.leave_one_out_reg(K_YY[:loo_size, :loo_size].cpu().numpy(),
                                                            labels=Kz[:loo_size, loo_size:].cpu().numpy(),
                                                            reg_list=reg_list)
            LOO_error[idx], under_tol, LOO_tol[idx] = \
                self.leave_one_out_reg_kernels(K_YY.cpu().numpy(), Kz.cpu().numpy(), reg_list)
            # if not np.any(under_tol)
            LOO_error[idx, under_tol] = 2.0 * LOO_error[idx].max()  # a hack to remove < tol lambdas
        LOO_idx = np.unravel_index(np.argmin(LOO_error, axis=None), LOO_error.shape)
        self.sigma2 = sigma2_list[LOO_idx[0]]
        self.ridge_lambda = reg_list[LOO_idx[1]]
        print('Best LOO parameters: sigma2 {}, lambda {}'.format(sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))
        if self.ridge_lambda < LOO_tol[LOO_idx[0]]:
            print('POORLY CONDITIONED MATRIX, switching lambda to SVD tolerance: {}'.format(LOO_tol[LOO_idx[0]]))
            self.ridge_lambda = LOO_tol[LOO_idx[0]]

        LOO_idx = np.unravel_index(np.argmin(LOO_error_sanity_check, axis=None), LOO_error_sanity_check.shape)
        print('Best LOO parameters (sanity check): sigma2 {}, lambda {}'.format(
            sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))

        print('LOO results\n{}'.format(LOO_error))
        print('LOO results (sanity check)\n{}'.format(LOO_error_sanity_check))

        return sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]

    def _get_yz_regressors(self, Y_heldout, Z_heldout, ky_sigma=None, kz_sigma=1.0):
        # self.yz_reg = defaultdict(dict)
        print('YZ regressors correction')
        # save memory
        # for mode in self.model_cfg.modes:
        #     del self.dataloaders[mode].dataset.linear_reg

        # train_modes = ['train', 'train_ood']
        # todo: add other modes
        # print('ONLY TRAIN/VAL HSIC IS CORRECT, OOD IS NOT')
        # for mode in train_modes:
        # try:
        # TODO: update heldout data
        # self.Y_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.targets_heldout)
        # self.Z_heldout = torch.FloatTensor(self.dataloaders[mode].dataset.distractors_heldout)

        print('Points saved')
        n_points = Y_heldout.shape[0]
        Kz = gaussian_kernel(Z_heldout, sigma2=kz_sigma)
        # eval(f'losses.{self.kernel_z}_kernel(self.Z_heldout, **self.kernel_z_args)')

        if self.loo_cond_mean:
            print('Estimating regressions parameters with LOO')
            reg_list = [1e-2, 1e-1, 1.0, 10.0, 100.0]
            sigma2_list = [10., 1.0, 0.1, 0.01]
            best_sigma2, best_reg = self._leave_one_out_regressors(Y_heldout, reg_list, sigma2_list, Kz)
        else:
            best_sigma2 = ky_sigma
            best_reg = self.ridge_lambda

        Ky = gaussian_kernel(Y_heldout, sigma2=best_sigma2)
        # Ky = eval(f'losses.{self.kernel_y}_kernel(self.Y_heldout, **self.kernel_y_args)')
        I = torch.eye(n_points, device=Ky.device)
        print('All gram matrices computed')
        W_all = torch.tensor(scp_solve(np.float128((Ky + best_reg * I).cpu().numpy()),
                                            np.float128(torch.cat((I, Kz), 1).cpu().numpy()),
                                            assume_a='pos')).float().to(Ky.device)
        print('W_all computed')
        print(f'** best sigma2 : {best_sigma2}, best ridge coeff : {best_reg}')
        W_1 = W_all[:, :n_points].to(self.device)
        W_2 = W_all[:, n_points:].to(self.device)

            # self.Y_heldout = self.Y_heldout.to(self.device)
            # self.Z_heldout = self.Z_heldout.to(self.device)
        # except:            
        #     W_1 = 0.
        #     W_2 = 0.
            # continue

        return W_1, W_2, best_sigma2  
                    

    def train(self, total_iteration=1e6, iter_per_valid=1000, batch_size=64, num_train=20000, num_heldout=2000, num_valid=2000, regularize_embedding=True):
        
        min_loss = 100000.
        max_score = -100000.
        best_weight = None
        mse_loss = nn.MSELoss()
        envname = self.envname

        heldout_data = self.replay_buffer_heldout.random_batch(num_heldout, standardize=self.standardize)
        expert_action_heldout = heldout_data['actions'][:, -self.action_dim:]
        expert_action_heldout = torch.tensor(expert_action_heldout, dtype=torch.float32, device=self.device)

        prev_expert_action_heldout = heldout_data['actions'][:, :-self.action_dim]
        prev_expert_action_heldout = torch.tensor(prev_expert_action_heldout, dtype=torch.float32, device=self.device)
        
        if self.action_standardize:
            Z_std_heldout = (prev_expert_action_heldout - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
            Y_std_heldout = (expert_action_heldout - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

            Z_std_heldout = Z_std_heldout.to(torch.float32)
            Y_std_heldout = Y_std_heldout.to(torch.float32)
        else:
            Z_std_heldout = prev_expert_action_heldout
            Y_std_heldout = expert_action_heldout           
        
        W_1, W_2, best_sigma2 = self._get_yz_regressors(Y_std_heldout, Z_std_heldout, kz_sigma=self.kz_sigma)
        
        batch_valid = self.replay_buffer_valid.random_batch(num_valid, standardize=self.standardize)        
        actions_valid = batch_valid['actions'][:, -self.action_dim:]
        prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim]
                
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)                        
                
        if self.action_standardize:
            Z_std_valid = (prev_expert_action_valid - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
            Y_std_valid = (actions_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

            Z_std_valid = Z_std.to(torch.float32)
            Y_std_valid = Y_std.to(torch.float32)
        else:
            Z_std_valid = prev_expert_action_valid
            Y_std_valid = actions_valid            
            
        circe_estimate_scaler = self.estimate_circe(X=Z_std_valid, 
                                                    Z=Z_std_valid, Z_heldout=Z_std_heldout,
                                                    Y=Y_std_valid, Y_heldout=Y_std_heldout, 
                                                    W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                    kx_sigma=self.kz_sigma, kz_sigma=self.kz_sigma)

            # W_1, W_2, best_sigma2 = self._get_yz_regressors(expert_action_heldout, prev_expert_action_heldout)
        
        
        for num in range(0, int(total_iteration)):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)

            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            prev_expert_action = batch['actions'][:, :-self.action_dim]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            prev_expert_action = torch.tensor(prev_expert_action, dtype=torch.float32, device=self.device)

            # NOTE:
            # obs = obs[:,obs_idx] -> processed in replay buffer
            # actions = torch.tanh(actions) -> processed in replay buffer          
            neg_likelihood = - self.policy.log_prob(obs, actions).mean()
            
            # NOTE:
            # estimate I(a^E_{t-1} ; a^I_{t}, a^E_{t})
            policy_action = self.policy(obs).rsample()
            # cur_action = torch.cat([actions, policy_action], dim=-1)

            if self.reg_coef > 0:                
                if self.action_standardize:
                    Z_std = (prev_expert_action - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                    Y_std = (actions - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                    Z_std = Z_std.to(torch.float32)
                    Y_std = Y_std.to(torch.float32)
                else:
                    Z_std = prev_expert_action
                    Y_std = actions
                        
                if regularize_embedding:
                    policy_embedding = self.policy.forward_embedding(obs)
                    circe_estimate = self.estimate_circe(X=policy_embedding, 
                                                         Z=Z_std, Z_heldout=Z_std_heldout,
                                                         Y=Y_std, Y_heldout=Y_std_heldout, 
                                                         W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                         kx_sigma=self.kx_sigma, kz_sigma=self.kz_sigma)
                else:
                    circe_estimate = self.estimate_circe(X=policy_action, 
                                                         Z=Z_std, Z_heldout=Z_std_heldout,
                                                         Y=Y_std, Y_heldout=Y_std_heldout, 
                                                         W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                         kx_sigma=self.kx_sigma, kz_sigma=self.kz_sigma)
            else:
                circe_estimate = 0.
            
            train_loss = neg_likelihood + self.reg_coef *  (1/circe_estimate_scaler) * circe_estimate 

            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num+1) % iter_per_valid == 0:

                batch_valid = self.replay_buffer_valid.random_batch(num_valid, standardize=self.standardize)
                obs_valid = batch_valid['observations']
                actions_valid = batch_valid['actions'][:, -self.action_dim:]
                prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim]
                        
                obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
                actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
                prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)                
                policy_action_valid = self.policy(obs_valid).sample()
                
                if self.action_standardize:
                    Z_std_valid = (prev_expert_action_valid - self.act_mean_tt[0, :-self.action_dim])/ self.act_std_tt[0, :-self.action_dim]
                    Y_std_valid = (actions_valid - self.act_mean_tt[0, -self.action_dim:])/ self.act_std_tt[0, -self.action_dim:]

                    Z_std_valid = Z_std.to(torch.float32)
                    Y_std_valid = Y_std.to(torch.float32)
                else:
                    Z_std_valid = prev_expert_action_valid
                    Y_std_valid = actions_valid

                if regularize_embedding:
                    policy_embedding_valid = self.policy.forward_embedding(obs_valid)
                    valid_circe_estimate = self.estimate_circe(X=policy_embedding_valid, 
                                                               Z=Z_std_valid, Z_heldout=Z_std_heldout,
                                                               Y=Y_std_valid, Y_heldout=Y_std_heldout, 
                                                               W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                               kx_sigma=self.kx_sigma, kz_sigma=self.kz_sigma)
                else:
                    valid_circe_estimate = self.estimate_circe(X=policy_action_valid, 
                                                               Z=Z_std_valid, Z_heldout=Z_std_heldout,
                                                               Y=Y_std_valid, Y_heldout=Y_std_heldout, 
                                                               W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                               kx_sigma=self.kx_sigma, kz_sigma=self.kz_sigma)
                    
                valid_circe_estimate_action = self.estimate_circe(X=policy_action_valid, 
                                                               Z=Z_std_valid, Z_heldout=Z_std_heldout,
                                                               Y=Y_std_valid, Y_heldout=Y_std_heldout, 
                                                               W_1=W_1, W_2=W_2, best_sigma2=best_sigma2,
                                                               kx_sigma=self.kx_sigma, kz_sigma=self.kz_sigma)

                valid_neg_likelihood = -self.policy.log_prob(obs_valid, actions_valid).mean()
                valid_loss = valid_neg_likelihood + self.reg_coef * (1/circe_estimate_scaler) * valid_circe_estimate

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num+1}: train_policy_loss={train_loss.item():.2f} ({obs.shape[0]}), val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'** iter{num+1}: train_mean_circe={(1/circe_estimate_scaler) * circe_estimate:.6f}, valid_mean_circe={(1/circe_estimate_scaler) * valid_circe_estimate:.6f} , valid_mean_circe(action)={(1/circe_estimate_scaler) * valid_circe_estimate_action:.6f}')
                
                if self.wandb:
                    self.wandb.log({'total_loss_train': train_loss.item(), 
                                    'total_loss_valid': valid_loss.item(),
                                    'train_neg_likelihood': neg_likelihood.item(),
                                    'valid_neg_likelihood': valid_neg_likelihood.item(),
                                    'train_mean_circe_estimate': (1/circe_estimate_scaler) * circe_estimate,
                                    'valid_mean_circe_estimate': (1/circe_estimate_scaler) * valid_circe_estimate,
                                    'valid_mean_circe(action)': (1/circe_estimate_scaler) * valid_circe_estimate_action,
                                    # 'MINE_loss_train': mine_loss.item(), 
                                    'eval_episode_return': eval_ret_mean
                                    }, step=num)

                if valid_loss < min_loss:
                # if eval_ret_mean > max_score:
                    print(f'** min loss! ')
                    min_loss = valid_loss
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
                    