import numpy as np
import gym


class Memory:
    def __init__(self, size, action_space):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        action_space: gym.spaces
            The action_space of sampled environment.
        """
        self._storage = []
        # TODO(zbzhu): Add self._traj_startpoints for convenience.
        self._maxsize = int(size)
        self._next_idx = 0
        self._discrete_act = isinstance(action_space, gym.spaces.Discrete)
        self._act_num = action_space.n if self._discrete_act else None

    def __len__(self):
        return len(self._storage)

    # TODO(zbzhu): Find a better way to merge two dict
    # without such get_<something> functions.
    def get_storage(self):
        return self._storage

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        if self._next_idx >= self._maxsize:
            self._next_idx = self._next_idx % self._maxsize
            # raise RuntimeError("Replay buffer size exceeded!")
        if self._discrete_act:
            one_hot_action = np.zeros(self._act_num)
            one_hot_action[action] = 1
            data = (obs_t, one_hot_action, reward, obs_tp1, done)
        else:
            data = (obs_t, action, reward, obs_tp1, done)
        self._storage.append(data)
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in idxes:
            data = self._storage[i]
            # obs_t, action, reward, obs_tp1, done = data
            obses_t.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            obses_tp1.append(data[3])
            dones.append(data[4])
        return {
            "observation": np.array(obses_t),
            "action": np.array(actions),
            "reward": np.array(rewards),
            "next_observation": np.array(obses_tp1),
            "done": np.array(dones),
        }

    def load_expert(self, data_dict, traj_num=None):
        if traj_num is None:
            traj_num = len(data_dict["observation"])
        else:
            assert traj_num >= len(data_dict["observation"])
        for ep_idx in range(traj_num):
            for obs, act, next_obs, done in zip(
                data_dict["observation"][ep_idx],
                data_dict["action"][ep_idx],
                data_dict["next_observation"][ep_idx],
                data_dict["done"][ep_idx],
            ):
                self.add(obs, act, None, next_obs, done)

    def make_index(self, batch_size):
        assert batch_size <= len(self._storage)
        return np.random.choice(len(self._storage), batch_size, replace=False)

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def append(self, memory):
        self._storage.extend(memory.get_storage())
