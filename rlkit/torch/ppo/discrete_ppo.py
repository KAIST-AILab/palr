from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Categorical

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DiscretePPOTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            vf,

            epsilon=0.05,
            reward_scale=1.0,

            lr=1e-3,
            optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.vf = vf

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.vf_criterion = nn.MSELoss()

        self.optimizer = optimizer_class(
            list(self.vf.parameters()) + list(self.policy.parameters()),
            lr=lr,
        )

        self.epsilon = epsilon
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.last_approx_kl = None

    def train_from_torch(self, batch):
        obs = batch['observations']
        old_log_pi = batch['log_prob']
        advantage = batch['advantage']
        returns = batch['returns']
        actions = batch['actions']

        """
        Policy Loss
        """
        _, _, all_probs = self.policy(obs)
        new_log_pi = torch.zeros(all_probs.shape[0], 1)
        for i in range(all_probs.shape[0]):
            new_log_pi[i] = Categorical(all_probs[i]).log_prob(actions[i]).sum()

        # Advantage Clip
        ratio = torch.exp(new_log_pi - old_log_pi)
        left = ratio * advantage
        right = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

        policy_loss = (-1 * torch.min(left, right)).mean()

        """
        VF Loss
        """
        v_pred = self.vf(obs)
        v_target = returns
        vf_loss = self.vf_criterion(v_pred, v_target)

        """
        Update networks
        """
        loss = policy_loss + vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.last_approx_kl is None or not self._need_to_update_eval_statistics:
            self.last_approx_kl = (old_log_pi - new_log_pi).detach()
        
        approx_ent = -new_log_pi

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            policy_grads = torch.cat([p.grad.flatten() for p in self.policy.parameters()])
            value_grads = torch.cat([p.grad.flatten() for p in self.vf.parameters()])

            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Target',
                ptu.get_numpy(v_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Gradients',
                ptu.get_numpy(policy_grads),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Value Gradients',
                ptu.get_numpy(value_grads),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy KL',
                ptu.get_numpy(self.last_approx_kl),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Entropy',
                ptu.get_numpy(approx_ent),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'New Log Pis',
                ptu.get_numpy(new_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Old Log Pis',
                ptu.get_numpy(old_log_pi),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.vf
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            vf=self.vf
        )
