import torch
from torch import nn
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.distributions import MultivariateDiagonalNormal, TanhNormal
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from torch.nn import functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', num_recurrent_layers=1):

        super().__init__()
        self.lstm_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            device=device
        )
        
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


class TanhGaussianPolicyPPOEmbedding(TorchStochasticPolicy):
    """
    Reference : 
    https://github.com/AlvinWen428/fighting-copycat-agents/blob/52dabfd8b1c42e50f31d84bd431915aad62e09cb/imitation_learning/models/gan_model/__init__.py#L9
    
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            embedding_dim,
            action_dim,
            policy_hidden_size,            
            policy_std=None,
            device='cpu',
            hidden_activation=nn.LeakyReLU,
            layer_norm=False,
            **kwargs
    ):
        if device =='cuda':
            ptu.set_gpu_mode(True)
        self.device = device
        
        super(TanhGaussianPolicyPPOEmbedding, self).__init__()
        
        self.input_size = embedding_dim
        self.output_size = action_dim
        self.hidden_activation = hidden_activation
        self.layer_norm = layer_norm

        self.embedding_params = []
        self.disc_params = []
        self.policy_params = []

        self.embed_fcs = []        

        self.policy_fcs = []
        
        self.disc_fcs = []
        
        self.device = device
        
        self.embedding_params = self.embed_fcs.parameters()

        self.policy_fcs = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(embedding_dim, policy_hidden_size, device=self.device),
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

    def forward(self, input):        
        h = self.policy_fcs(input)
        policy_mean = self.policy_mean(h)

        if self.policy_std is None:
            policy_log_std = self.policy_fc_log_std(h)
            policy_log_std = torch.clamp(policy_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            policy_std = torch.exp(policy_log_std)
        else:
            policy_std = torch.from_numpy(np.array([self.policy_std, ])).float().to(ptu.device)

        return TanhNormal(policy_mean, policy_std)

    def logprob_mean_std(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def logprob_input_action(self, input, action):
        tanh_normal = self.forward(input)
        log_prob = tanh_normal.log_prob(
            action,
        )        
        return log_prob

    def predict_action(self, input, deterministic=True):
        tanh_normal = self.forward(input)
        if deterministic:
            pred_action = tanh_normal.mean
        else:
            pred_action = tanh_normal.sample()
        return pred_action
