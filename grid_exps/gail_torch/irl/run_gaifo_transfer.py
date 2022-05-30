import os
import torch
import pickle
import gym
import yaml
import numpy as np
import math
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from gail_torch.utils import Memory, make_env
from gail_torch.policy import PPOPolicy, DiscretePPOPolicy, Discriminator
from gail_torch.sampler import Sampler


def KL(p, q):
    # p,q为两个list，里面存着对应的取值的概率，整个list相加为1
    if 0 in q:
        raise ValueError
    return sum(_p * math.log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)


def JS(p, q):
    M = [0.5 * (_p + _q) for (_p, _q) in zip(p, q)]
    return 0.5 * (KL(p, M) + KL(q, M))


def compute_jsd(obs_a, obs_b):
    dist_a = obs_a.sum(axis=0)
    dist_a = dist_a / dist_a.sum()
    dist_b = obs_b.sum(axis=0)
    dist_b = dist_b / dist_b.sum()
    idx = np.where(dist_a + dist_b != 0)[0]
    return JS(dist_a[idx], dist_b[idx])


def do_train(config):
    """
    init the env, agent and train the agents
    """
    env = make_env(
        config["env_specs"]["env_name"],
        repeat_ratio=config["env_specs"]["repeat_ratio"],
    )
    env.seed(config["exp_params"]["seed"])
    torch.manual_seed(config["exp_params"]["seed"])
    print("=============================")
    print("=1 env {} is right ...".format(config["env_specs"]["env_name"]))
    print("=============================")

    expert_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    with open(config["basic"]["expert_path"], "rb") as f:
        expert_data_dict = pickle.load(f)
    expert_memory.load_expert(expert_data_dict)
    print("=2 Loading dataset with {} samples ...".format(len(expert_memory)))
    print("=============================")

    if config["others"]["use_tensorboard_log"]:
        log_dir = os.path.join(
            config["basic"]["tb_log_dir"],
            config["basic"]["exp_name"],
            datetime.now().strftime("%Y.%m.%d.%H.%M.%S"),
        )
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    discriminator = Discriminator(
        observation_space=env.observation_space,
        action_space=env.action_space,
        writer=writer,
        lr=config["exp_params"]["disc_lr"],
        expert_memory=expert_memory,
        state_only=config["exp_params"]["state_only"],
        device=config["basic"]["device"],
    ).to(config["basic"]["device"])

    if isinstance(env.action_space, gym.spaces.Discrete):
        print("=3 using discrete ppo policy ...")
        policy = DiscretePPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            actor_lr=config["exp_params"]["actor_lr"],
            critic_lr=config["exp_params"]["critic_lr"],
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    elif isinstance(env.action_space, gym.spaces.Box):
        print("=3 using continuous ppo policy ...")
        policy = PPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            actor_lr=config["exp_params"]["actor_lr"],
            critic_lr=config["exp_params"]["critic_lr"],
            activation="tanh",
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    else:
        raise NotImplementedError
    print("=============================")

    model_save_dir = os.path.join(
        config["basic"]["model_save_dir"], config["basic"]["exp_name"]
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    memory = Memory(config["exp_params"]["memory_size"], action_space=env.action_space)
    eval_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )

    print("=4 starting iterations ...")
    print("=============================")
    sampler = Sampler(
        env,
        policy,
        memory,
        config["basic"]["device"],
        writer=writer,
        num_threads=config["exp_params"]["num_threads"],
        clear_memory=True,
    )

    eval_sampler = Sampler(
        env,
        policy,
        eval_memory,
        config["basic"]["device"],
        writer=writer,
        clear_memory=True,
        num_threads=config["exp_params"]["num_threads"],
    )

    tot_metrics = {"jsd": [], "expert_density": [], "agent_density": [], "avg_rew": []}
    for update_cnt in range(config["exp_params"]["num_updates"] + 1):

        if (
            config["others"]["model_save_freq"] > 0
            and update_cnt > 0
            and update_cnt % config["others"]["model_save_freq"] == 0
        ):
            torch.save(
                policy,
                os.path.join(model_save_dir, f"model_{update_cnt}.pkl"),
            )
            print("=model saved at episode {}".format(update_cnt))

        memory, _, _ = sampler.collect_samples(config["exp_params"]["update_timestep"])
        policy.update(memory)
        discriminator.update(memory)

        if update_cnt % 10 == 0:
            memory, _, avg_rew = eval_sampler.collect_samples(1024)
            agent_batch = memory.collect()
            agent_obs = agent_batch["observation"]
            expert_batch = expert_memory.collect()
            expert_obs = expert_batch["observation"]
            agent_density = agent_obs.sum(axis=0)
            agent_density = agent_density / agent_density.sum()
            expert_density = expert_obs.sum(axis=0)
            expert_density = expert_density / expert_density.sum()
            jsd = compute_jsd(agent_obs, expert_obs) * 10
            tot_metrics["jsd"].append(jsd)
            tot_metrics["avg_rew"].append(avg_rew)
            tot_metrics["expert_density"].append(expert_density)
            tot_metrics["agent_density"].append(agent_density)

    import gc

    del policy, discriminator
    gc.collect()

    env = make_env(
        config["env_specs"]["env_name"],
        repeat_ratio=config["env_specs"]["repeat_ratio"] * 4,
    )
    env.seed(config["exp_params"]["seed"])
    memory = Memory(config["exp_params"]["memory_size"], action_space=env.action_space)
    eval_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    sampler.env = env
    eval_sampler.env = env
    sampler.memory = memory
    eval_sampler.memory = eval_memory

    discriminator = Discriminator(
        observation_space=env.observation_space,
        action_space=env.action_space,
        writer=writer,
        lr=config["exp_params"]["disc_lr"],
        expert_memory=expert_memory,
        state_only=config["exp_params"]["state_only"],
        device=config["basic"]["device"],
    ).to(config["basic"]["device"])

    if isinstance(env.action_space, gym.spaces.Discrete):
        print("=3 using discrete ppo policy ...")
        policy = DiscretePPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            actor_lr=config["exp_params"]["actor_lr"],
            critic_lr=config["exp_params"]["critic_lr"],
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    elif isinstance(env.action_space, gym.spaces.Box):
        print("=3 using continuous ppo policy ...")
        policy = PPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            actor_lr=config["exp_params"]["actor_lr"],
            critic_lr=config["exp_params"]["critic_lr"],
            activation="tanh",
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    else:
        raise NotImplementedError
    sampler.policy = policy
    eval_sampler.policy = policy

    print("============================")
    print("\n Start transfer!!!!!!\n")
    print("============================")

    for update_cnt in range(config["exp_params"]["num_updates"] + 1):

        if (
            config["others"]["model_save_freq"] > 0
            and update_cnt > 0
            and update_cnt % config["others"]["model_save_freq"] == 0
        ):
            torch.save(
                policy,
                os.path.join(model_save_dir, f"model_{update_cnt}.pkl"),
            )
            print("=model saved at episode {}".format(update_cnt))

        memory, _, _ = sampler.collect_samples(config["exp_params"]["update_timestep"])
        policy.update(memory)
        discriminator.update(memory)

        if update_cnt % 10 == 0:
            memory, _, avg_rew = eval_sampler.collect_samples(1024)
            agent_batch = memory.collect()
            agent_obs = agent_batch["observation"]
            expert_batch = expert_memory.collect()
            expert_obs = expert_batch["observation"]
            agent_density = agent_obs.sum(axis=0)
            agent_density = agent_density / agent_density.sum()
            expert_density = expert_obs.sum(axis=0)
            expert_density = expert_density / expert_density.sum()
            jsd = compute_jsd(agent_obs, expert_obs) * 10
            tot_metrics["avg_rew"].append(avg_rew)
            tot_metrics["jsd"].append(jsd)
            tot_metrics["expert_density"].append(expert_density)
            tot_metrics["agent_density"].append(agent_density)

    # plot_metrics(tot_metrics)
    with open(
        "gaifo_{}_seed{}.pkl".format(
            config["env_specs"]["repeat_ratio"], config["exp_params"]["seed"]
        ),
        "wb",
    ) as f:
        pickle.dump(tot_metrics, f)

    # torch.save(
    #     policy,
    #     os.path.join(model_save_dir, "model_final.pkl"),
    # )
    print("=model saved at episode {}".format(update_cnt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GAIfO for single agent environments")
    parser.add_argument(
        "--exp_config",
        "-e",
        help="path to the experiment configuration yaml file",
        default="./config/gail.yaml",
    )
    parser.add_argument(
        "--use_tensorboard_log",
        "-t",
        action="store_true",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
    )
    args = parser.parse_args()

    with open(args.exp_config, "rb") as f:
        config = yaml.load(f)

    config["others"]["use_tensorboard_log"] = args.use_tensorboard_log
    if args.use_tensorboard_log:
        print("\nUSING TENSORBOARD LOG\n")

    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device("cuda")
        print("USING GPU\n")
    else:
        print("USING CPU\n")
        device = torch.device("cpu")
    config["basic"]["device"] = device

    os.makedirs(config["basic"]["model_save_dir"], exist_ok=True)
    if config["others"]["use_tensorboard_log"] and not os.path.exists(
        config["basic"]["tb_log_dir"]
    ):
        os.makedirs(config["basic"]["tb_log_dir"])

    print(config, "\n")

    do_train(config)
