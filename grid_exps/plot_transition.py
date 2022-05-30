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


# visited_loc = np.array(
#     [
#         [0, 0],
#         [0, 1],
#         [0, 2],
#         [0, 3],
#         [0, 4],
#         [1, 0],
#         [1, 5],
#         [2, 0],
#         [2, 5],
#         [3, 0],
#         [3, 1],
#         [3, 2],
#         [3, 3],
#         [3, 4],
#         [3, 5],
#     ]
# )

visited_loc = []
for x in range(6):
    for y in range(6):
        visited_loc.append([x, y])
visited_loc = np.array(visited_loc)


if __name__ == "__main__":
    model_path = "./gail_torch/assets/learned_models/dpo/model_final.pkl"
    expert_path = "./gail_torch/assets/expert_traj/expert.pkl"

    env = gym.make("GridWorld-v0")
    policy = torch.load(model_path).to("cpu")
    policy.random_action = False
    policy_sp = policy.state_predictor

    transition_heatgraph = []
    for loc in visited_loc:
        loc_idx = loc[0] * env.map_size[0] + loc[1]
        obs = one_hot_tensor(loc_idx, env.num_map_loc)
        next_obs_logits = policy_sp(obs)
        next_obs_probs = _t2n(torch.softmax(next_obs_logits, dim=-1).unsqueeze(-1))
        transition_heatgraph.append(gradient_color(next_obs_probs))
    transition_heatgraph = np.stack(transition_heatgraph)

    with open(expert_path, "rb") as f:
        expert_data = pickle.load(f)

    expert_obs = expert_data["observation"]
    expert_next_obs = expert_data["next_observation"]

    expert_transition_cnt = np.zeros((env.num_map_loc, env.num_map_loc))
    for ep_obs, ep_next_obs in zip(expert_obs, expert_next_obs):
        for obs, next_obs in zip(ep_obs, ep_next_obs):
            expert_transition_cnt[obs.argmax()] += next_obs

    expert_heatgraph = []
    for loc in visited_loc:
        loc_idx = loc[0] * env.map_size[0] + loc[1]
        expert_heatgraph.append(
            gradient_color(
                np.expand_dims(expert_transition_cnt[loc_idx], axis=-1)
                / expert_transition_cnt[loc_idx].sum()
            )
        )
    expert_heatgraph = np.stack(expert_heatgraph)

    fig, axes = plt.subplots(2)

    axes[0].imshow(expert_heatgraph / 255)
    axes[0].set_title("Expert")
    axes[0].set_xlabel("state prime")
    axes[0].set_ylabel("state")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(transition_heatgraph / 255)
    axes[1].set_title("DPO")
    axes[1].set_xlabel("state prime")
    axes[1].set_ylabel("state")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.subplots_adjust(hspace=0.4)

    plt.savefig("state_predictor.pdf", dpi=300, bbox_inches="tight")
