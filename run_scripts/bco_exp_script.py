import yaml
import argparse
import joblib
import numpy as np
import os, sys, inspect
import pickle, random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs.wrappers import ScaledEnv, EPS
from rlkit.torch.common.policies import (
    ReparamTanhMultivariateGaussianPolicy,
    InverseDynamic,
)
from rlkit.torch.algorithms.irl.bco import BCO
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.safe_load(f.read())
    # expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    # buffer_save_dict = joblib.load(expert_demos_path)
    # expert_replay_buffer = buffer_save_dict['train']
    demos_path = listings[variant["expert_name"]]["file_paths"][0]
    print("demos_path", demos_path)
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    traj_list = random.sample(traj_list, variant["traj_num"])

    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
    acts = np.vstack([traj_list[i]["actions"] for i in range(len(traj_list))])
    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
    # acts_mean, acts_std = np.mean(acts, axis=0), np.std(acts, axis=0)
    acts_mean, acts_std = None, None
    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])
    training_env = get_env(env_specs)
    training_env.seed(env_specs["training_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    expert_replay_buffer = EnvReplayBuffer(
        variant["bco_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")
        env = ScaledEnv(
            env=env,
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        training_env = ScaledEnv(
            env=training_env,
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_mean) / (
                obs_std + EPS
            )
            traj_list[i]["next_observations"] = (
                traj_list[i]["next_observations"] - obs_mean
            ) / (obs_std + EPS)

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(traj_list[i], env=env)

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    inverse_dynamics = InverseDynamic(
        hidden_sizes=num_hidden * 2 * [net_size],
        input_size=obs_dim * 2,
        output_size=action_dim,
    )

    algorithm = BCO(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        inverse_dynamics=inverse_dynamics,
        expert_replay_buffer=expert_replay_buffer,
        **variant["bco_params"]
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["num_gpu_per_worker"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
