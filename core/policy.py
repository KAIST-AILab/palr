import torch
from torch import nn
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.distributions import MultivariateDiagonalNormal, TanhNormal
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from torch.nn import functional as F
# torch.set_default_dtype(torch.float64)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            device='cpu',
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            device=device,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        
        if device =='cuda':
            ptu.set_gpu_mode(True)
        

        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim, device=self.device)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)

    def log_prob(self, obs, action):
        normal = self.forward(obs)
        log_prob = normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob


class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            device='cpu',
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            device=device,
            **kwargs
        )
        self.log_std = None
        self.std = std
        
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim, device=self.device)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def forward_with_h(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std), h

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob(self, obs, action):
        tanh_normal = self.forward(obs)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

class TanhGaussianPolicyWithDiscriminator(TorchStochasticPolicy):
    """
    Reference : 
    https://github.com/AlvinWen428/fighting-copycat-agents/blob/52dabfd8b1c42e50f31d84bd431915aad62e09cb/imitation_learning/models/gan_model/__init__.py#L9
    
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            embedding_dim,
            embedding_hidden_size,
            policy_hidden_size,
            disc_hidden_size,
            policy_std=None,
            disc_std=None,
            init_w=1e-3,
            device='cpu',
            hidden_activation=F.leaky_relu,
            use_discriminator_action_input=True,
            layer_norm=False,
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super(TanhGaussianPolicyWithDiscriminator, self).__init__()
        #     hidden_sizes,
        #     input_size=obs_dim,
        #     output_size=action_dim,
        #     init_w=init_w,
        #     device=device,
        #     **kwargs
        # )

        self.input_size = obs_dim
        self.output_size = action_dim
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.embedding_params = []
        self.disc_params = []
        self.policy_params = []

        self.embed_fcs = []
        # self.embed_layer_norms = []

        self.policy_fcs = []
        # self.policy_layer_norms = []

        self.disc_fcs = []
        # self.disc_layer_norms = []
        
        self.device = device
        in_size = self.input_size

        # self.embed_fc1 = nn.Linear(in_size, embedding_hidden_size, device=self.device)
        # self.embed_fc1.weight.data.uniform_(-init_w, init_w)
        # self.embed_fc1.bias.data.fill_(0)
        # self.embedding_params.append({'params': self.embed_fc1.parameters()})

        # self.embed_fc2 = nn.Linear(embedding_hidden_size, embedding_dim, device=self.device)
        # self.embed_fc2.weight.data.uniform_(-init_w, init_w)
        # self.embed_fc2.bias.data.fill_(0)
        # self.embedding_params.append({'params': self.embed_fc2.parameters()})
        self.embed_fcs = nn.Sequential(
            nn.Linear(self.input_size, embedding_hidden_size, bias=False, device=self.device),
            # nn.BatchNorm1d(embedding_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_hidden_size, embedding_dim, device=self.device)
        )
        self.embedding_params = self.embed_fcs.parameters()

        self.policy_fcs = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(embedding_dim, policy_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.policy_params.append({'params': self.policy_fcs.parameters()})
        self.policy_mean = nn.Linear(policy_hidden_size, action_dim, device=self.device)
        self.policy_params.append({'params': self.policy_mean.parameters()})        
        
        # self.policy_fc1 = nn.Linear(embedding_dim, policy_hidden_size, device=self.device)
        # self.policy_fc1.weight.data.uniform_(-init_w, init_w)
        # self.policy_fc1.bias.data.fill_(0)
        # self.policy_params.append({'params': self.policy_fc1.parameters()})        
        # self.policy_fc2 = nn.Linear(policy_hidden_size, action_dim, device=self.device)
        # self.policy_fc2.weight.data.uniform_(-init_w, init_w)
        # self.policy_fc2.bias.data.fill_(0)
        # self.policy_params.append({'params': self.policy_fc2.parameters()})        

        self.policy_log_std = None
        self.policy_std = policy_std
        
        if policy_std is None:
            self.policy_fc_log_std = nn.Linear(policy_hidden_size, action_dim, device=self.device)
            # self.policy_fc_log_std.weight.data.uniform_(-init_w, init_w)
            # self.policy_fc_log_std.bias.data.uniform_(-init_w, init_w)
            self.policy_params.append({'params': self.policy_fc_log_std.parameters()})
        else:
            self.policy_log_std = np.log(policy_std)
            assert LOG_SIG_MIN <= self.policy_log_std <= LOG_SIG_MAX

        self.use_discriminator_action_input = use_discriminator_action_input
        if use_discriminator_action_input:
            self.discriminator_input_dim = embedding_dim + action_dim
        else:
            self.discriminator_input_dim = embedding_dim

        self.disc_fcs = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.discriminator_input_dim, disc_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disc_params.append({'params': self.disc_fcs.parameters()})
        self.disc_mean = nn.Linear(disc_hidden_size, action_dim, device=self.device)
        self.disc_params.append({'params': self.disc_mean.parameters()})
        # self.disc_fc1 = nn.Linear(self.discriminator_input_dim, disc_hidden_size, device=self.device)
        # self.disc_fc1.weight.data.uniform_(-init_w, init_w)
        # self.disc_fc1.bias.data.fill_(0)
        # self.disc_params.append({'params': self.disc_fc1.parameters()})
        # self.disc_fcs.append(self.disc_fc1)

        # self.disc_fc2 = nn.Linear(disc_hidden_size, action_dim, device=self.device)
        # self.disc_fc2.weight.data.uniform_(-init_w, init_w)
        # self.disc_fc2.bias.data.fill_(0)
        # self.disc_params.append({'params': self.disc_fc2.parameters()})

        self.disc_log_std = None
        self.disc_std = disc_std
        
        if disc_std is None:
            self.disc_fc_log_std = nn.Linear(disc_hidden_size, action_dim, device=self.device)
            # self.disc_fc_log_std.weight.data.uniform_(-init_w, init_w)
            # self.disc_fc_log_std.bias.data.uniform_(-init_w, init_w)
            self.disc_params.append({'params': self.disc_fc_log_std.parameters()})
        else:
            self.disc_log_std = np.log(disc_std)
            assert LOG_SIG_MIN <= self.disc_log_std <= LOG_SIG_MAX        

    def forward(self, obs):
        # h = obs

        # h = self.hidden_activation(self.embed_fc1(h))
        # h = self.embed_fc2(h)

        # h = self.hidden_activation(self.policy_fc1(h))
        # policy_mean = self.policy_fc2(h)

        h = self.embed_fcs(obs)
        h = self.policy_fcs(h)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def forward_embedding(self, obs):
        # h = obs
        
        # h = self.hidden_activation(self.embed_fc1(h))
        # h = self.embed_fc2(h)
        h = self.embed_fcs(obs)

        return h

    def forward_discriminator_from_embedding(self, h, current_action):
        if self.use_discriminator_action_input:
            disc_h = torch.cat((h, current_action), 1)
        else:
            disc_h = h
        
        # disc_h = self.hidden_activation(disc_h)
        # h = self.embed_fcs(obs)
        # h = self.policy_fcs(h)
        # disc_h = self.hidden_activation(self.disc_fc1(disc_h))

        disc_h = self.disc_fcs(disc_h)
        disc_mean = self.disc_mean(disc_h)

        if self.disc_std is None:
            disc_log_std = self.disc_fc_log_std(disc_h)
            disc_log_std = torch.clamp(disc_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            disc_std = torch.exp(disc_log_std)
        else:
            disc_std = torch.from_numpy(np.array([self.disc_std, ])).float().to(ptu.device)

        return TanhNormal(disc_mean, disc_std)

    def forward_policy_from_embedding(self, h):
        # h = self.hidden_activation(h)
        # h = self.hidden_activation(self.policy_fc1(h))
        h = self.policy_fcs(h)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def forward_both(self, obs, current_action):
        # h = obs

        # h = self.hidden_activation(self.embed_fc1(h))
        # h = self.embed_fc2(h)
        h = self.embed_fcs(obs)

        if self.use_discriminator_action_input:
            disc_h = torch.cat((h, current_action), 1)
        else:
            disc_h = h

        # h = self.hidden_activation(self.policy_fc1(h))
        h = self.policy_fcs(h)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        # disc_h = self.hidden_activation(self.disc_fc1(disc_h))
        disc_h = self.disc_fcs(disc_h)
        disc_mean = self.disc_mean(disc_h)

        if self.disc_std is None:
            disc_log_std = self.disc_fc_log_std(disc_h)
            disc_log_std = torch.clamp(disc_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            disc_std = torch.exp(disc_log_std)
        else:
            disc_std = torch.from_numpy(np.array([self.disc_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std), TanhNormal(disc_mean, disc_std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob_both(self, obs, current_action, prev_action):
        tanh_normal_policy, tanh_normal_disc = self.forward_both(obs, current_action)
        
        policy_log_prob = tanh_normal_policy.log_prob(
            current_action,
        )
        disc_log_prob = tanh_normal_disc.log_prob(
            prev_action,
        )

        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return policy_log_prob, disc_log_prob

    def log_prob(self, obs, action):
        tanh_normal = self.forward(obs)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob_discriminator(self, obs, current_action, target_action):
        tanh_normal = self.forward_discriminator(obs, current_action)
        log_prob = tanh_normal.log_prob(
            target_action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob_policy_from_embedding(self, h, action):
        tanh_normal = self.forward_policy_from_embedding(h)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def predict_action_from_embedding(self, h):
        tanh_normal = self.forward_policy_from_embedding(h)
        pred_action = tanh_normal.mean            
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return pred_action

    def predict_prev_action_from_embedding_current_a(self, h, current_action):
        tanh_normal = self.forward_discriminator_from_embedding(h, current_action)
        pred_action = tanh_normal.mean            
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return pred_action

    def log_prob_discriminator_from_embedding(self, h, current_action, target_action):
        tanh_normal = self.forward_discriminator_from_embedding(h, current_action)
        log_prob = tanh_normal.log_prob(
            target_action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

class TanhGaussianRAPPolicy(TorchStochasticPolicy):
    """
    Reference : 
    https://github.com/AlvinWen428/fighting-copycat-agents/blob/52dabfd8b1c42e50f31d84bd431915aad62e09cb/imitation_learning/models/gan_model/__init__.py#L9
    
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            obs_dim,
            stack_size,
            action_dim,
            embedding_dim,
            embedding_hidden_size,
            policy_hidden_size,
            residual_hidden_size,
            policy_std=None,
            residual_std=0.1,
            device='cpu',
            hidden_activation=F.leaky_relu,            
            layer_norm=False,
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super(TanhGaussianRAPPolicy, self).__init__()
        
        self.input_size = obs_dim
        self.stack_size = stack_size
        self.output_size = action_dim
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.embedding_params = []
        self.residual_params = []
        self.policy_params = []

        self.history_embed_fcs = []
        self.single_embed_fcs = []
        # self.embed_layer_norms = []

        self.policy_fcs = []
        # self.policy_layer_norms = []

        self.residual_fcs = []
        # self.residual_layer_norms = []
        
        self.device = device
        in_size = self.input_size

        self.history_embed_fcs = nn.Sequential(
            nn.Linear(self.input_size * self.stack_size, embedding_hidden_size, bias=False, device=self.device),
            # nn.BatchNorm1d(embedding_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_hidden_size, embedding_dim, device=self.device)
        )
        self.history_embedding_params = self.history_embed_fcs.parameters()
        
        self.single_embed_fcs = nn.Sequential(
            nn.Linear(self.input_size, embedding_hidden_size, bias=False, device=self.device),
            # nn.BatchNorm1d(embedding_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_hidden_size, embedding_dim, device=self.device)
        )
        self.single_embedding_params = self.single_embed_fcs.parameters()

        self.policy_fcs = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(embedding_dim*2, policy_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.policy_params.append({'params': self.policy_fcs.parameters()})
        self.policy_mean = nn.Linear(policy_hidden_size, action_dim, device=self.device)
        self.policy_params.append({'params': self.policy_mean.parameters()})             

        self.policy_log_std = None
        self.policy_std = policy_std
        
        if policy_std is None:
            self.policy_fc_log_std = nn.Linear(policy_hidden_size, action_dim, device=self.device)
            # self.policy_fc_log_std.weight.data.uniform_(-init_w, init_w)
            # self.policy_fc_log_std.bias.data.uniform_(-init_w, init_w)
            self.policy_params.append({'params': self.policy_fc_log_std.parameters()})
        else:
            self.policy_log_std = np.log(policy_std)
            assert LOG_SIG_MIN <= self.policy_log_std <= LOG_SIG_MAX

        self.residual_fcs = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(embedding_dim, residual_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual_params.append({'params': self.residual_fcs.parameters()})
        self.residual_mean = nn.Linear(residual_hidden_size, action_dim, device=self.device)        
        self.residual_params.append({'params': self.residual_mean.parameters()})
        
        # self.residual_log_std = None
        # self.residual_std = residual_std
        
        # if residual_std is None:
        #     self.residual_fc_log_std = nn.Linear(residual_hidden_size, action_dim, device=self.device)
        #     self.residual_params.append({'params': self.residual_fc_log_std.parameters()})
        # else:
        #     self.residual_log_std = np.log(residual_std)
        #     assert LOG_SIG_MIN <= self.residual_log_std <= LOG_SIG_MAX        

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = obs[None]
            
        obs_total = obs
        obs_current = obs[:, -self.input_size:]

        m = self.history_embed_fcs(obs_total)
        h = self.single_embed_fcs(obs_current)                
        
        policy_input = torch.cat([m.detach(), h], dim=-1)
        
        policy_input = self.policy_fcs(policy_input)
        policy_mean = self.policy_mean(policy_input)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(policy_input)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        policy_dist = TanhNormal(policy_mean, policy_std)
        #residual_dist = TanhNormal(residual_mean, residual_std)
        
        return policy_dist #, residual_dist

    def forward_embedding(self, obs):
        obs_total = obs
        obs_current = obs[:, -self.input_size:]

        m = self.history_embed_fcs(obs_total)
        h = self.single_embed_fcs(obs_current)

        return m, h

    def forward_residual_from_m(self, m):
        residual_m = self.residual_fcs(m)
        residual_mean = self.residual_mean(residual_m)
        
        # if self.residual_std is None:
        #     residual_log_std = self.residual_fc_log_std(residual_m)
        #     residual_log_std = torch.clamp(residual_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     residual_std = torch.exp(residual_log_std)
        # else:
        #     residual_std = torch.from_numpy(np.array([self.residual_std, ])).float().to(ptu.device)
            
        # residual_dist = TanhNormal(residual_mean, residual_std)

        return residual_mean

    def forward_policy_from_embedding(self, m, h):
        policy_input = torch.cat([m.detach(), h], dim=-1)
        
        policy_input = self.policy_fcs(policy_input)
        policy_mean = self.policy_mean(policy_input)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(policy_input)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob
    
    def log_prob(self, obs, action):
        tanh_normal = self.forward(obs)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    # def log_prob_residual_from_m(self, m, action_residual):
    #     tanh_normal = self.forward_residual_from_m(m)
    #     log_prob = tanh_normal.log_prob(
    #         action_residual,
    #     )
    #     # log_prob = log_prob.sum(dim=1, keepdim=True)
    #     return log_prob

    def log_prob_policy_from_m_h(self, m, h, action):        
        tanh_normal = self.forward_policy_from_embedding(m, h)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def predict_action_from_m_h(self, m, h):
        tanh_normal = self.forward_policy_from_embedding(m, h)
        pred_action = tanh_normal.mean            
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return pred_action
    

class TanhGaussianPolicyWithEmbedding(TorchStochasticPolicy):
    """
    Reference : 
    https://github.com/AlvinWen428/fighting-copycat-agents/blob/52dabfd8b1c42e50f31d84bd431915aad62e09cb/imitation_learning/models/gan_model/__init__.py#L9
    
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            embedding_dim,
            embedding_hidden_size,
            policy_hidden_size,            
            policy_std=None,
            disc_std=None,
            init_w=1e-3,
            device='cpu',
            hidden_activation=F.leaky_relu,            
            layer_norm=False,
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super(TanhGaussianPolicyWithEmbedding, self).__init__()
        #     hidden_sizes,
        #     input_size=obs_dim,
        #     output_size=action_dim,
        #     init_w=init_w,
        #     device=device,
        #     **kwargs
        # )

        self.input_size = obs_dim
        self.output_size = action_dim
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.embedding_params = []
        self.disc_params = []
        self.policy_params = []

        self.embed_fcs = []
        # self.embed_layer_norms = []

        self.policy_fcs = []
        # self.policy_layer_norms = []

        self.disc_fcs = []
        # self.disc_layer_norms = []
        
        self.device = device
        in_size = self.input_size

        self.embed_fcs = nn.Sequential(
            nn.Linear(self.input_size, embedding_hidden_size, bias=False, device=self.device),
            # nn.BatchNorm1d(embedding_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_hidden_size, embedding_dim, device=self.device),            
        )
        self.embedding_params = self.embed_fcs.parameters()

        self.policy_fcs = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(embedding_dim, policy_hidden_size, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # self.policy_params.append({'params': self.policy_fcs.parameters()})
        self.policy_mean = nn.Linear(policy_hidden_size, action_dim, device=self.device)
        self.policy_params.append({'params': self.policy_mean.parameters()})        
        
        # self.policy_fc1 = nn.Linear(embedding_dim, policy_hidden_size, device=self.device)
        # self.policy_fc1.weight.data.uniform_(-init_w, init_w)
        # self.policy_fc1.bias.data.fill_(0)
        # self.policy_params.append({'params': self.policy_fc1.parameters()})        
        # self.policy_fc2 = nn.Linear(policy_hidden_size, action_dim, device=self.device)
        # self.policy_fc2.weight.data.uniform_(-init_w, init_w)
        # self.policy_fc2.bias.data.fill_(0)
        # self.policy_params.append({'params': self.policy_fc2.parameters()})        

        self.policy_log_std = None
        self.policy_std = policy_std
        
        if policy_std is None:
            self.policy_fc_log_std = nn.Linear(policy_hidden_size, action_dim, device=self.device)
            # self.policy_fc_log_std.weight.data.uniform_(-init_w, init_w)
            # self.policy_fc_log_std.bias.data.uniform_(-init_w, init_w)
            self.policy_params.append({'params': self.policy_fc_log_std.parameters()})
        else:
            self.policy_log_std = np.log(policy_std)
            assert LOG_SIG_MIN <= self.policy_log_std <= LOG_SIG_MAX

    def forward(self, obs):
        # h = obs

        # h = self.hidden_activation(self.embed_fc1(h))
        # h = self.embed_fc2(h)

        # h = self.hidden_activation(self.policy_fc1(h))
        # policy_mean = self.policy_fc2(h)

        h = self.embed_fcs(obs)
        h = self.policy_fcs(h)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def forward_embedding(self, obs):
        # h = obs
        
        # h = self.hidden_activation(self.embed_fc1(h))
        # h = self.embed_fc2(h)
        h = self.embed_fcs(obs)

        return h

    def forward_policy_from_embedding(self, h):
        # h = self.hidden_activation(h)
        # h = self.hidden_activation(self.policy_fc1(h))
        h = self.policy_fcs(h)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob(self, obs, action):
        tanh_normal = self.forward(obs)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob_policy_from_embedding(self, h, action):
        tanh_normal = self.forward_policy_from_embedding(h)
        log_prob = tanh_normal.log_prob(
            action,
        )
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def predict_action_from_embedding(self, h):
        tanh_normal = self.forward_policy_from_embedding(h)
        pred_action = tanh_normal.mean            
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        return pred_action
