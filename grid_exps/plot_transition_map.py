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
        np.array([255, 255, 255]),
        axis=0,
    ).repeat(repeat_num, 0)
    end = np.expand_dims(
        np.array([251, 200, 47]),
        axis=0,
    ).repeat(repeat_num, 0)
    return start + prob.dot(end[None, 0] - start[None, 0])


def _t2n(tensor):
    return tensor.detach().cpu().numpy()


def data_to_img(data, img_shape=None):
    data = data / data.max()
    img_shape = [360, 360, 3]
    img = np.zeros(img_shape, dtype=np.int64)
    gs0 = int(img.shape[0] / data.shape[0])
    gs1 = int(img.shape[1] / data.shape[1])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1] = gradient_color(
                data[i, j].reshape(1, -1)
            )[0].astype(int)
    return img


visited_loc = np.array([[i, j] for i in range(6) for j in range(6)])
visited_loc = visited_loc[:-1]


if __name__ == "__main__":
    # for method_name in ['dpo_sup', 'dpo', 'dpo-weighted', None]:

    for method_name in ["dpo-weighted"]:

        model_path = "./gail_torch/assets/learned_models/{}/model_final.pkl".format(
            method_name
        )
        expert_path = "./gail_torch/assets/expert_traj/expert.pkl"

        env = gym.make("GridWorld-v0")
        # try:
        policy = torch.load(model_path).to("cpu")
        policy.random_action = False
        policy_sp = policy.state_predictor
        # except:
        #     pass

        with open(expert_path, "rb") as f:
            expert_data = pickle.load(f)

        expert_density = np.array(expert_data["observation"]) + np.array(
            expert_data["next_observation"]
        )
        expert_density = (expert_density.sum(axis=0).sum(axis=0) > 0).astype(float)
        # expert_density = expert_density / expert_density.sum()
        print(expert_density)
        expert_heatmap = data_to_img(expert_density.reshape(6, 6))
        plt.imshow(expert_heatmap)

        img_shape = [360, 360, 3]
        grid_x = img_shape[0] / env.map_size[0]
        grid_y = img_shape[1] / env.map_size[1]

        # if method_name is not None:
        if False:

            for loc in visited_loc:
                loc_idx = loc[0] * env.map_size[0] + loc[1]
                obs = one_hot_tensor(loc_idx, env.num_map_loc)
                next_obs_logits = policy_sp(obs)
                # next_obs_probs = _t2n(torch.softmax(next_obs_logits, dim=-1).unsqueeze(-1))
                next_obs_idx = next_obs_logits.argmax()
                next_loc = np.array(
                    (next_obs_idx // env.map_size[0], next_obs_idx % env.map_size[0])
                )

                loc = np.array([loc[1], loc[0]])
                next_loc = np.array([next_loc[1], next_loc[0]])

                gs0 = int(img_shape[0] / env.map_size[0])
                gs1 = int(img_shape[1] / env.map_size[1])
                loc_img = (loc + 1 / 2) * [gs0, gs1]
                next_loc_img = (next_loc + 1 / 2) * [gs0, gs1]
                delta_loc_img = (next_loc_img - loc_img) - 20 * (
                    next_loc_img - loc_img
                ) / np.linalg.norm(next_loc_img - loc_img)
                plt.arrow(*loc_img, *delta_loc_img, width=1, head_width=5)

        else:
            print("None")
            # plt.plot([grid_x*6, grid_x*3], [grid_y*3, grid_y*3], color='k', linestyle="--", linewidth=1)
            # plt.plot([grid_x*3, grid_x*3], [0, grid_y*3], color='k', linestyle="--", linewidth=1)
            plt.plot(
                [grid_x * 6, grid_x * 3],
                [grid_y * 3, grid_y * 3],
                color="k",
                linestyle="--",
                linewidth=1,
            )
            plt.plot(
                [grid_x * 3, grid_x * 3],
                [grid_y * 6, grid_y * 3],
                color="k",
                linestyle="--",
                linewidth=1,
            )
            # plt.gca().add_patch(plt.Rectangle((grid_x*1,grid_y*6),grid_x*3,grid_y*3, color='g',alpha=0.5))
            plt.fill(
                [grid_x * 6, grid_x * 3, grid_x * 3, grid_x * 6],
                [grid_y * 3, grid_y * 3, grid_y * 6, grid_y * 6],
                facecolor="g",
                alpha=0.5,
            )

            for loc in visited_loc:
                loc_idx = loc[0] * env.map_size[0] + loc[1]
                if expert_density[loc_idx] == 0:
                    continue
                obs = one_hot_tensor(loc_idx, env.num_map_loc)
                next_obs_logits = policy_sp(obs)
                # next_obs_probs = _t2n(torch.softmax(next_obs_logits, dim=-1).unsqueeze(-1))
                next_obs_idx = next_obs_logits.argmax()
                next_loc = np.array(
                    (next_obs_idx // env.map_size[0], next_obs_idx % env.map_size[0])
                )

                loc = np.array([loc[1], loc[0]])
                next_loc = np.array([next_loc[1], next_loc[0]])

                gs0 = int(img_shape[0] / env.map_size[0])
                gs1 = int(img_shape[1] / env.map_size[1])
                loc_img = (loc + 1 / 2) * [gs0, gs1]
                next_loc_img = (next_loc + 1 / 2) * [gs0, gs1]
                delta_loc_img = (next_loc_img - loc_img) - 20 * (
                    next_loc_img - loc_img
                ) / np.linalg.norm(next_loc_img - loc_img)
                plt.arrow(*loc_img, *delta_loc_img, width=1, head_width=5)

        # plt.savefig("transition_map.png", dpi=300, bbox_inches='tight')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(xmin=0, xmax=img_shape[0])
        plt.ylim(ymin=0, ymax=img_shape[1])
        # plt.savefig("transition_map_{}.pdf".format(method_name), dpi=300, bbox_inches='tight')
        plt.savefig("transition_map_hack.pdf", dpi=300, bbox_inches="tight")
        plt.clf()
