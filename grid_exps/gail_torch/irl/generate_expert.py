import gym
import time
import gym_gridworld
import numpy as np
import pickle


action_sets = [[0, 3, 0, 3, 0, 3, 0, 3, 0, 3]]

env = gym.make("GridWorld-v0")
ep_num = 100
tot_obs = []
tot_act = []
tot_next_obs = []
tot_rew = []
tot_done = []
for ep in range(ep_num):
    action_set = action_sets[0]
    obs = env.reset()
    ep_obs = []
    ep_act = []
    ep_next_obs = []
    ep_rew = []
    ep_done = []
    for episode_step in range(1000):
        act = action_set[episode_step]

        next_obs, rew, done, info = env.step(act)

        ep_obs.append(obs)
        ep_act.append(act)
        ep_rew.append(rew)
        ep_next_obs.append(next_obs)
        ep_done.append(done)

        if done:
            # print(episode_step, np.argmax(obs), rew)
            break

        obs = next_obs

    tot_obs.append(np.array(ep_obs))
    tot_act.append(np.array(ep_act))
    tot_rew.append(np.array(ep_rew))
    tot_next_obs.append(np.array(ep_next_obs))
    tot_done.append(np.array(ep_done))

expert_dict = {
    "observation": tot_obs,
    "next_observation": tot_next_obs,
    "action": tot_act,
    "done": tot_done,
}

with open("../../gail_torch/assets/expert_traj/expert.pkl", "wb") as f:
    pickle.dump(expert_dict, f)
