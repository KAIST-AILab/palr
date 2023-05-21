from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class PPOEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._log_prob = np.zeros((max_replay_buffer_size, 1))
        self._advantage = np.zeros((max_replay_buffer_size, 1))
        self._return = np.zeros((max_replay_buffer_size, 1))

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info,
                advantage,
                returns,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
            path["advantages"],
            path["returns"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                advantage=advantage,
                returns=returns
            )
        self.terminate_episode()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, agent_info, advantage, returns, **kwargs):
        """
        Log Probability of action is stored in agent_info
        Empty Advantage is stored
        """
        self._log_prob[self._top] = agent_info["log_prob"]
        self._advantage[self._top] = advantage
        self._return[self._top] = returns

        return super().add_sample(
            observation=observation, 
            action=action,
            reward=reward, 
            terminal=terminal,
            next_observation=next_observation,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            log_prob=self._log_prob[indices],
            advantage=self._advantage[indices],
            returns=self._return[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def end_epoch(self, epoch):
        self._observations.fill(0)
        self._next_obs.fill(0)
        self._actions.fill(0)
        self._rewards.fill(0)
        self._log_prob.fill(0)
        self._advantage.fill(0)
        self._return.fill(0)
        self._terminals.fill(0)
        for key in self._env_infos.keys():
            self._env_infos[key].fill(0)
        self._top = 0
        self._size = 0
