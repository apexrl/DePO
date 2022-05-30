import torch
import pickle
import gym
import gym_gridworld

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def one_hot_tensor(loc, tot_loc):
    t = torch.zeros(tot_loc)
    t[loc] = 1
    return t


def gradient_color(prob: float):
    repeat_num = prob.shape[0]
    start = np.expand_dims(
        np.array([22, 97, 171]),
        axis=0,
    ).repeat(repeat_num, 0)
    end = np.expand_dims(
        np.array([251, 200, 47]),
        axis=0,
    ).repeat(repeat_num, 0)
    return start + prob.dot(end[None, 0] - start[None, 0])


def _t2n(tensor):
    return tensor.detach().cpu().numpy()


visited_loc = np.array(
    [
        [0, 0],
        [0, 3],
        [3, 5],
    ]
)


if __name__ == "__main__":
    model_path = "/Users/zhuzhengbang/Code_Repos/lfo_grid/gail_torch/assets/learned_models/dpo/model_final.pkl"
    expert_path = "/Users/zhuzhengbang/Code_Repos/lfo_grid/gail_torch/assets/expert_traj/expert.pkl"

    env = gym.make("GridWorld-v0", repeat_ratio=5)
    act_num = env.action_space.n
    policy = torch.load(model_path)
    policy.random_action = False
    policy.deterministic_action = False
    policy_inv = policy.inverse_dynamic
    policy_sp = policy.state_predictor

    with open(expert_path, "rb") as f:
        expert_data = pickle.load(f)

    expert_obs = expert_data["observation"]
    expert_act = expert_data["action"]

    expert_act_cnt = np.zeros((env.num_map_loc, act_num))
    for ep_obs, ep_act in zip(expert_obs, expert_act):
        for obs, act in zip(ep_obs, ep_act):
            expert_act_cnt[obs.argmax()] += _t2n(one_hot_tensor(act * 5 + 2, act_num))

    expert_histgraph = []
    for loc in visited_loc:
        loc_idx = loc[0] * env.map_size[0] + loc[1]
        expert_histgraph.append(expert_act_cnt[loc_idx] / expert_act_cnt[loc_idx].sum())
    expert_histgraph = np.concatenate(expert_histgraph, axis=0)

    agent_histgraph = []
    for loc in visited_loc:
        loc_idx = loc[0] * env.map_size[0] + loc[1]
        one_hot_obs = one_hot_tensor(loc_idx, env.num_map_loc)
        next_obs_logits = policy_sp(one_hot_obs)
        next_obs_probs = torch.softmax(next_obs_logits, axis=-1)
        accum_prob = torch.zeros(act_num)
        for idx, prob in enumerate(next_obs_probs):
            one_hot_next_obs = one_hot_tensor(idx, env.num_map_loc)
            _input = torch.cat([one_hot_obs, one_hot_next_obs])
            act_probs = torch.softmax(policy_inv(_input), dim=-1)
            accum_prob += act_probs * prob
        agent_histgraph.append(_t2n(accum_prob / accum_prob.sum()))
    agent_histgraph = np.concatenate(agent_histgraph, axis=0)

    fig, axes = plt.subplots(2)

    axes[0].bar(
        np.arange(len(expert_histgraph)),
        height=expert_histgraph,
    )
    axes[0].vlines([19.5, 39.5], ymin=0, ymax=1, linestyles="dashed", colors="r")
    # axes[0].set_title("Expert")
    # axes[0].set_xlabel("action index")
    axes[0].set_ylabel("selection probability")
    axes[0].set_xticks([])
    # axes[0].set_xticks([0, 19.5, 39.5, 60])
    # axes[0].set_xticklabels(["0", "20/0", "20/0", "20"])

    axes[1].bar(
        np.arange(len(agent_histgraph)),
        height=agent_histgraph,
    )
    axes[1].vlines(
        [19.5, 39.5], ymin=0, ymax=max(agent_histgraph), linestyles="dashed", colors="r"
    )
    # axes[1].set_title("DPO")
    # axes[1].set_xlabel("action index")
    axes[1].set_ylabel("selection probability")
    axes[1].set_xticks([])
    # axes[1].set_xticks([0, 19.5, 39.5, 60])
    # axes[1].set_xticklabels(["0", "20/0", "20/0", "20"])

    plt.subplots_adjust(hspace=0.3)

    plt.savefig("inverse_dynamic.pdf", dpi=300, bbox_inches="tight")
