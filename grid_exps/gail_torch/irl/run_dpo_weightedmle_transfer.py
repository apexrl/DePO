import os
import torch
import pickle
import numpy as np
import yaml
import math
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from gail_torch.utils import Memory, make_env
from gail_torch.policy import PPOPolicy, DiscretePPOPolicy, Discriminator, DPO_W
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


def plot_metrics(metrics):
    fig = plt.figure()
    for key in metrics.keys():
        plt.plot(np.arange(len(metrics[key])), metrics[key], label=key)
    plt.ylim(0, 5)
    plt.legend()
    fig.savefig("dpo.png")


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

    epsilon = config["exp_params"]["epsilon"]

    discriminator = Discriminator(
        observation_space=env.observation_space,
        action_space=env.action_space,
        writer=writer,
        lr=config["exp_params"]["disc_lr"],
        expert_memory=expert_memory,
        state_only=config["exp_params"]["state_only"],
        device=config["basic"]["device"],
    ).to(config["basic"]["device"])

    # sp_discriminator = Discriminator(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     writer=writer,
    #     lr=config["exp_params"]["disc_lr"],
    #     expert_memory=expert_memory,
    #     state_only=True,
    #     device=config["basic"]["device"],
    # ).to(config["basic"]["device"])

    print("=3 using dpo ...")
    policy = DPO_W(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_memory=expert_memory,
        writer=writer,
        discriminator=discriminator,
        device=config["basic"]["device"],
        **config["exp_params"]["policy_params"],
    ).to(config["basic"]["device"])
    print("=============================")

    model_save_dir = os.path.join(
        config["basic"]["model_save_dir"], config["basic"]["exp_name"]
    )
    os.makedirs(model_save_dir, exist_ok=True)

    train_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    big_memory = Memory(1000000, action_space=env.action_space)
    eval_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )

    print("=4 starting iterations ...")
    print("=============================")
    sampler = Sampler(
        env,
        policy,
        train_memory,
        config["basic"]["device"],
        writer=writer,
        clear_memory=True,
        num_threads=config["exp_params"]["num_threads"],
        additional_memory=big_memory,
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
    policy.random_action = False
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

        memory, inv_memory, _ = sampler.collect_samples(
            config["exp_params"]["update_timestep"], epsilon=epsilon
        )
        metrics = policy.update_inv(inv_memory)
        policy.update_policy(memory)
        discriminator.update(memory)

        if update_cnt % 10 == 0:
            # for key in metrics.keys():
            #     if key in tot_metrics:
            #         tot_metrics[key].append(metrics[key])
            #     else:
            #         tot_metrics[key] = [metrics[key]]
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

    env = make_env(
        config["env_specs"]["env_name"],
        repeat_ratio=config["env_specs"]["repeat_ratio"] * 4,
    )
    env.seed(config["exp_params"]["seed"])
    train_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    big_memory = Memory(1000000, action_space=env.action_space)
    eval_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    policy.reset_inverse_dynamics(act_num=env.action_space.n)
    sampler.env = env
    eval_sampler.env = env
    sampler.memory = train_memory
    sampler.additional_memory = big_memory
    eval_sampler.memory = eval_memory

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

        memory, inv_memory, _ = sampler.collect_samples(
            config["exp_params"]["update_timestep"], epsilon=epsilon
        )
        metrics = policy.update_inv(inv_memory)
        # policy.update_policy(memory)
        # discriminator.update(memory)

        if update_cnt % 10 == 0:
            # for key in metrics.keys():
            #     if key in tot_metrics:
            #         tot_metrics[key].append(metrics[key])
            #     else:
            #         tot_metrics[key] = [metrics[key]]
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
        "dpo_{}_seed{}.pkl".format(
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
    parser = argparse.ArgumentParser("DPO for single agent environments")
    parser.add_argument(
        "--exp_config",
        "-e",
        help="path to the experiment configuration yaml file",
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

    if config["others"]["use_tensorboard_log"] and not os.path.exists(
        config["basic"]["tb_log_dir"]
    ):
        os.makedirs(config["basic"]["tb_log_dir"])

    print(config, "\n")

    do_train(config)
