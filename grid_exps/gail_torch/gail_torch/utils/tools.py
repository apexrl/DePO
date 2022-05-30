import torch
import numpy as np
import gym
import gym_gridworld
from ray import tune


def make_env(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    return env


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, dones, values, gamma, tau, device):
    rewards, dones, values = to_device(torch.device("cpu"), rewards, dones, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * (1 - dones[i]) - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * (1 - dones[i])

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def flat_dict_to_episode(input_dict):
    """Convert flat data to episode-devided data.

    The input_dict may contains multiple keys, such as 'state',
    'action', etc. It must include 'done' to indicate
    the end of a episode. Initially, value of each key is
    a flat array with shape like (batch_size, [agent_num], ...).
    This function is aimed at converting every value into the
    shape of (batch_size, episode_len, [agent_num], ...) according
    to the 'done' info. In case of episodes with variable length,
    zero padding is adopted.

    Args:
        input_dict: A dict with 'done' in its keys.

    Returns:
        A converted dict and a torch.tensor of shape
        (batch_size, ) describing the orignal length
        of each episode before padding.
    """
    if len(input_dict["done"].shape) == 2:  # multi-agent
        dones = torch.all(input_dict["done"] > 0, axis=1)
    else:
        dones = input_dict["done"] > 0

    _device = dones.device

    return_dict = {}

    # The trajectory i has the range of [traj_start_points[i], traj_endpoints[i]].
    traj_endpoints = torch.where(dones > 0)[0]
    traj_startpoints = torch.roll(traj_endpoints, 1, 0) + 1
    traj_startpoints[0] = 0
    traj_len = traj_endpoints - traj_startpoints + 1
    max_traj_len = traj_len.max()

    for key in input_dict.keys():
        if len(input_dict[key]) == 0:
            return_dict[key] = np.array([])
            continue
        flat_arr = input_dict[key]
        devided_arr = []
        for idx in range(len(traj_endpoints)):
            traj_arr = flat_arr[traj_startpoints[idx] : traj_endpoints[idx] + 1]
            cur_len = traj_endpoints[idx] - traj_startpoints[idx] + 1
            traj_arr = torch.cat(
                [
                    traj_arr,
                    torch.zeros(
                        (max_traj_len - cur_len, *flat_arr[0].shape), device=_device
                    ),
                ]
            )
            devided_arr.append(traj_arr)
        return_dict[key] = torch.stack(devided_arr)

    return return_dict, traj_len


def generate_tune_config(config):
    variable = config["variable"]
    config = _merge_variable_to_config(config, variable)
    return config


def _merge_variable_to_config(config, variable):
    for k in variable.keys():
        if isinstance(variable[k], dict):
            if k not in config:
                config[k] = {}
            config[k] = _merge_variable_to_config(config[k], variable[k])
        elif isinstance(variable[k], list):
            config[k] = tune.grid_search(variable[k])
        else:
            raise RuntimeError("The variable part of tune config file is broken.")
    return config
