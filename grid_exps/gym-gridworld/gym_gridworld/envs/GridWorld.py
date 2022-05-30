import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# define colors
# 0: white; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {
    0: [1.0, 1.0, 1.0],
    1: [0.5, 0.5, 0.5],
    2: [0.0, 0.0, 1.0],
    3: [0.0, 1.0, 0.0],
    4: [1.0, 0.0, 0.0],
    6: [1.0, 0.0, 1.0],
    7: [1.0, 1.0, 0.0],
}


def row_in_mat(row, mat):
    return len(np.where((mat == row).all(1))[0]) != 0


class GridWorld(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_step=20,
        map_size=[6, 6],
        reach_reward=1.0,
        repeat_ratio=1,
    ):
        self.img_shape = [360, 360, 3]
        self.start_pos = np.array([0, 0])
        self.target_pos = np.array([5, 5])
        self.map_size = np.array(map_size)
        self.reach_reward = reach_reward
        self.num_map_loc = self.map_size[0] * self.map_size[1]
        self.episode_step = 0
        self.max_step = max_step
        self.num_actions = 4
        self._obs_high = np.array([1.0 for _ in range(self.num_map_loc)])
        self._obs_low = np.array([0.0 for _ in range(self.num_map_loc)])
        self.action_space = spaces.Discrete(self.num_actions * repeat_ratio)
        self.observation_space = spaces.Box(low=self._obs_low, high=self._obs_high)
        self.done = False
        self.move_offset = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        self.repeat_ratio = repeat_ratio
        self.state = self.start_pos

    def seed(self, seed):
        np.random.seed(seed)

    def _get_state(self):
        one_hot_state = np.zeros(self.num_map_loc)
        one_hot_state[self.state[0] * self.map_size[0] + self.state[1]] = 1
        return one_hot_state

    def reset(self):
        # self.state = self.start_pos
        start_x = np.random.randint(6)
        if start_x > 2:
            start_y = np.random.randint(3)
        else:
            start_y = np.random.randint(6)
        self.state = np.array([start_x, start_y])
        self.done = False
        self.episode_step = 0
        return self._get_state()

    def step(self, action):
        assert self.action_space.contains(action), "wrong action"
        assert not self.done, "try to step in a finished environment"
        info = {}
        self.state = self._take_action(action)
        # print(self.state, self.state[0] * self.map_size[0] + self.state[1])
        one_hot_state = np.zeros(self.map_size[0] * self.map_size[1])
        one_hot_state[self.state[0] * self.map_size[0] + self.state[1]] = 1
        info["one_hot_state"] = one_hot_state
        reward = self._get_reward()
        self.episode_step += 1
        if self.episode_step == self.max_step or (self.state == self.target_pos).all():
            self.done = True
        return self._get_state(), reward, self.done, info

    def _gridmap_to_img(self, grid_map, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        img = np.zeros(img_shape, dtype=np.float32)
        gs0 = int(img.shape[0] / grid_map.shape[0])
        gs1 = int(img.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                img[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1] = np.array(
                    COLORS[grid_map[i, j]]
                )
        return img

    def _data_to_img(self, data, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        img = np.zeros(img_shape, dtype=np.float32)
        gs0 = int(img.shape[0] / data.shape[0])
        gs1 = int(img.shape[1] / data.shape[1])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                img[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1] = np.array(
                    [1 - data[i, j], 1 - data[i, j], 1 - data[i, j]]
                )
        return img

    def render(self, mode="human"):
        assert mode in ["human", "image"]
        grid_map = np.zeros(shape=self.map_size, dtype=int)
        grid_map[self.state[0], self.state[1]] = 2
        grid_map[self.target_pos[0], self.target_pos[1]] = 4
        img = self._gridmap_to_img(grid_map)
        if mode == "human":
            fig = plt.figure(0)
            plt.clf()
            plt.imshow(img)
            fig.canvas.draw()
            plt.pause(0.00001)
            return
        elif mode == "image":
            return img

    def _is_out_map(self, state):
        if (
            state[0] < 0
            or state[0] >= self.map_size[0]
            or state[1] < 0
            or state[1] >= self.map_size[1]
        ):
            return True
        else:
            return False

    def _take_action(self, action):
        action = action // self.repeat_ratio
        cur_state = self.state
        if self._is_out_map(cur_state + self.move_offset[action]):
            new_state = cur_state
        else:
            new_state = cur_state + self.move_offset[action]
        return new_state

    def _get_reward(self):
        if (self.state == self.target_pos).all():
            return self.reach_reward
        else:
            return 0
