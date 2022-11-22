import abc
import numpy as np

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        traj_end_indices = np.sort(list(np.nonzero(path["timeouts"])[0]) + list(np.nonzero(path["terminals"])[0]))
        if 'next_observations' not in path.keys():
            observations = np.delete(path["observations"][:-1,:], traj_end_indices, axis=0)
            next_observations = np.delete(path["observations"][1:,:], traj_end_indices, axis=0)

            actions = np.delete(path["actions"][:-1, :], traj_end_indices, axis=0)
            rewards = np.delete(path["rewards"][:-1], traj_end_indices)
            timeouts = np.delete(path["timeouts"][:-1], traj_end_indices)         # do not shift one previous step
            terminals = np.delete(path["terminals"][:-1], traj_end_indices)       # do not shift one previous step
            
            n_data = observations.shape[0]
            if 'env_infos' not in path.keys():
                env_infos = np.array([{} for _ in range(n_data)])
            else:
                env_infos = np.delete(path["env_infos"][:-1], traj_end_indices)

            if 'agent_infos' not in path.keys():
                agent_infos = np.array([{} for _ in range(n_data)])
            else:
                agent_infos = np.delete(path["agent_infos"][:-1], traj_end_indices)

            terminals = np.logical_or(timeouts, terminals)

            for i, (
                    obs,
                    action,
                    reward,
                    next_obs,
                    terminal,
                    agent_info,
                    env_info
            ) in enumerate(zip(
                observations,
                actions,
                rewards,
                next_observations,
                terminals,
                agent_infos,
                env_infos,
            )):
                self.add_sample(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
            self.terminate_episode()

        else:
            for i, (
                    obs,
                    action,
                    reward,
                    next_obs,
                    terminal,
                    agent_info,
                    env_info
            ) in enumerate(zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
            )):
                self.add_sample(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    terminal=terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
            self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return

