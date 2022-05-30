import yaml
import argparse
import joblib
import numpy as np
import os, sys, inspect
import pickle, random
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import eval_util

from rlkit.envs.wrappers import ScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.common.policies import MakeDeterministic

from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianLfOPolicy

from video import save_image, save_video


def experiment(variant, lfo):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    test_env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])
    test_env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    if variant["scale_env_with_demo_stats"]:
        with open("demos_listing.yaml", "r") as f:
            listings = yaml.safe_load(f.read())
        expert_demos_path = listings[variant["expert_name"]]["file_paths"][
            variant["expert_idx"]
        ]
        print("demos_path", expert_demos_path)
        with open(expert_demos_path, "rb") as f:
            traj_list = pickle.load(f)
        traj_list = random.sample(traj_list, variant["traj_num"])

        obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
        acts = np.vstack([traj_list[i]["actions"] for i in range(len(traj_list))])
        obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
        # acts_mean, acts_std = np.mean(acts, axis=0), np.std(acts, axis=0)
        acts_mean, acts_std = None, None
        obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)
        env = ScaledEnv(
            env,
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        test_env = ScaledEnv(
            test_env,
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )

    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    assert lfo, "should test over lfo models!"

    policy = joblib.load(variant["policy_checkpoint"])["exploration_policy"][0]
    state_predictor = policy.state_predictor
    inverse_dynamic = policy.inverse_dynamic

    if variant["eval_deterministic"]:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    pred_images = []
    images = []
    image = None
    render = variant["render"]
    render_kwargs = variant["render_kwargs"]
    render_mode = variant["render_mode"]

    init_observation = env.reset()
    _ = test_env.reset()
    pred_obs_prime = None

    start_step = variant["start_step"]

    observation = init_observation
    test_pred_mse = []
    obs_sequence = []
    for _ in range(1000):
        obs_sequence.append(observation)

        if render:
            if render_mode == "rgb_array":
                image = env.render(mode=render_mode, **render_kwargs)
                images.append(image)
            else:
                env.render(**render_kwargs)

        pred_obs_prime, action, agent_info = policy.get_action(
            obs_np=observation, return_predicting_obs=True
        )

        next_ob, reward, terminal, env_info = env.step(action)
        if terminal:
            print("Terminal！", _)
            break
        try:
            error = np.mean(
                (env.get_unscaled_obs(next_ob) - env.get_unscaled_obs(pred_obs_prime))
                ** 2
            )
        except:
            error = np.mean((next_ob - pred_obs_prime) ** 2)
        test_pred_mse.append(error)
        observation = next_ob
    print("pred_mse:", np.mean(test_pred_mse))

    observation = obs_sequence[start_step]
    # import mujoco_py
    # test_env.viewer = mujoco_py.MjRenderContextOffscreen(test_env.sim, -1)
    # renderer_kwargs = {
    #     'trackbodyid': 2,
    #     'distance': 3,
    #     'lookat': [0, -0.5, 1],
    #     'elevation': -20
    # }

    # for key, val in renderer_kwargs.items():
    #     if key == 'lookat':
    #         test_env.viewer.cam.lookat[:] = val[:]
    #     else:
    #         setattr(test_env.viewer.cam, key, val)
    for _ in range(exp_specs["plan_step"]):
        pred_obs_prime = observation
        try:
            pred_obs_prime = test_env.get_unscaled_obs(pred_obs_prime)
        except:
            pass

        qpos_shape = env.sim.data.qpos.shape[0]
        qvel_shape = env.sim.data.qvel.shape[0]
        qpos = np.concatenate(
            [[test_env.sim.data.qpos[0]], pred_obs_prime[: qpos_shape - 1]]
        )
        qvel = pred_obs_prime[-qvel_shape:]
        if "invpendulum" in variant["env_specs"]["env_name"]:
            qpos = pred_obs_prime[:qpos_shape]
        if "humanoid" in variant["env_specs"]["env_name"]:
            qpos = np.concatenate(
                [test_env.sim.data.qpos[0:2], pred_obs_prime[: qpos_shape - 2]]
            )
        # elif "walker"  in variant["env_specs"]["env_name"]:

        if render:
            try:
                test_env.set_state(qpos, qvel)
            except:
                break
            if render_mode == "rgb_array":
                image = test_env.render(mode=render_mode, **render_kwargs)
                pred_images.append(image)
            else:
                test_env.render(**render_kwargs)
        pred_obs_prime, action, agent_info = policy.get_action(
            obs_np=observation, return_predicting_obs=True
        )

        observation = pred_obs_prime

    # fps = 1 // getattr(env, "dt", 1 / 30)
    len_pred = len(pred_images)
    fps = variant["fps"]
    if variant["render"] and variant["render_mode"] == "rgb_array":
        video_path = variant["video_path"]
        img_path = variant["img_path"]
        video_path = os.path.join(video_path, variant["env_specs"]["env_name"])
        img_path = os.path.join(img_path, variant["env_specs"]["env_name"], "pred")

        print("saving videos...")
        pred_images = np.stack(pred_images, axis=0)
        video_save_path = os.path.join(video_path, f"episode_pred.mp4")
        save_video(pred_images, video_save_path, fps=fps)
        save_image(pred_images, img_path, start_frame=variant["start_step"])

        video_path = variant["video_path"]
        img_path = variant["img_path"]
        video_path = os.path.join(video_path, variant["env_specs"]["env_name"])
        img_path = os.path.join(img_path, variant["env_specs"]["env_name"], "real")

        print("saving videos...")
        # fps = 30
        images = np.stack(images, axis=0)
        video_save_path = os.path.join(video_path, f"episode_real.mp4")
        save_video(images[start_step : start_step + len_pred], video_save_path, fps=fps)
        save_image(images, img_path, start_frame=0)

    return


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    parser.add_argument(
        "-s", "--save_res", help="save result to file", type=int, default=1
    )

    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    if exp_specs["num_gpu_per_worker"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    # setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    train_file = ""
    pkl_name = "/best.pkl"
    exp_specs["scale_env_with_demo_stats"] = True
    exp_specs["expert_name"] = exp_specs["env_specs"]["env_name"] + "_sac"
    exp_specs["start_step"] = 0
    exp_specs["fps"] = 30

    if "dpo" in exp_specs["method"]:
        if "humanoid" in exp_specs["env_specs"]["env_name"]:

            # train_file = 'dpo-humanoidslim-unionsp-mlemse-staticalpha-0.0-weightedmle-norm-normal-gail2--state-diff--biginvbuffer-unionsp--gp-16.0--spalpha-0.01--idlr-0.0001--rs-2.0--inviter-0--invevery-10/'
            # _file = 'dpo_humanoidslim_unionsp_mlemse_staticalpha_0.0_weightedmle_norm_normal_gail2--state_diff--biginvbuffer-unionsp--gp-16.0--spalpha-0.01--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_11_08_46_02_0001--s-4/'
            # exp_specs["seed"] = 4

            train_file = "dpo-humanoidslim-unionsp-mlemse-staticalpha-0.0-weightedmle-norm-minmax-gail2--state-diff--biginvbuffer-unionsp--gp-16.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            _file = "dpo_humanoidslim_unionsp_mlemse_staticalpha_0.0_weightedmle_norm_minmax_gail2--state_diff--biginvbuffer-unionsp--gp-16.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_09_07_24_36_0001--s-0/"

            pkl_name = "/params.pkl"
            exp_specs["expert_name"] = exp_specs["env_specs"]["env_name"] + "_slim_sac"
            # pkl_name = "/best.pkl"

            exp_specs["fps"] = 5

            exp_specs["start_step"] = 80
            exp_specs["plan_step"] = 30

        if "hopper" in exp_specs["env_specs"]["env_name"]:
            # train_file = "dpo-hopper-mlemse-staticalpha-0.0-norm-weightedmle-valid-gail2-rewshape-q-weight--state-diff--biginvbuffer-unionsp--gp-4.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            # _file = "dpo_hopper_mlemse_staticalpha_0.0_norm_weightedmle_valid_gail2_rewshape_q_weight--state_diff--biginvbuffer-unionsp--gp-4.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_03_06_19_50_0000--s-0/"
            train_file = "dpo-hopper-mlemse-staticalpha-0.0-norm-weightedmle-valid-gail2-rewshape-q-norm-clip-exps--state-diff--biginvbuffer-unionsp--gp-4.0--spalpha-1.0--idlr-0.0001/"
            _file = "dpo_hopper_mlemse_staticalpha_0.0_norm_weightedmle_valid_gail2_rewshape_q_norm_clip_exps--state_diff--biginvbuffer-unionsp--gp-4.0--spalpha-1.0--idlr-0.0001_2021_12_26_10_58_26_0000--s-0/"
            pkl_name = "/params.pkl"

            exp_specs["start_step"] = 90
            exp_specs["plan_step"] = 300

        elif "walker" in exp_specs["env_specs"]["env_name"]:
            train_file = "dpo-walker-union-mlemse-staticalpha-0.0-weightedmle-validinv-norm-weight--state-diff--biginvbuffer-unionsp--gp-8.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            _file = "dpo_walker_union_mlemse_staticalpha_0.0_weightedmle_validinv_norm_weight--state_diff--biginvbuffer-unionsp--gp-8.0--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_02_11_39_36_0000--s-0/"
            pkl_name = "/best.pkl"
            exp_specs["fps"] = 5

            exp_specs["start_step"] = 90
            exp_specs["plan_step"] = 40
        elif "halfcheetah" in exp_specs["env_specs"]["env_name"]:
            # train_file = "dpo-halfcheetah-unionsp-mlemse-staticalpha-0.0-weightedmle-norm-minmax--state-diff--biginvbuffer-unionsp--gp-0.5--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            # _file = 'dpo_halfcheetah_unionsp_mlemse_staticalpha_0.0_weightedmle_norm_minmax--state_diff--biginvbuffer-unionsp--gp-0.5--spalpha-0.1--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_04_12_13_37_0000--s-0'
            train_file = "dpo-halfcheetah-unionsp-mlemse-staticalpha-0.0-weightedmle-norm-minmax--state-diff--biginvbuffer-unionsp--gp-0.5--spalpha-0.01--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            _file = "dpo_halfcheetah_unionsp_mlemse_staticalpha_0.0_weightedmle_norm_minmax--state_diff--biginvbuffer-unionsp--gp-0.5--spalpha-0.01--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_04_20_43_13_0000--s-0/"
            pkl_name = "/best.pkl"
            exp_specs["fps"] = 5

            exp_specs["start_step"] = 90
            exp_specs["plan_step"] = 20
        elif "ant" in exp_specs["env_specs"]["env_name"]:
            train_file = ""
        elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "dpo-invpendulum-mlemse-minmax-lesspretrain--state-diff--biginvbuffer-unionsp--gp-4.0--spalpha-1.0--idlr-0.0001--rs-2.0--inviter-0--invevery-10/"
            _file = "dpo_invpendulum_mlemse_minmax_lesspretrain--state_diff--biginvbuffer-unionsp--gp-4.0--spalpha-1.0--idlr-0.0001--rs-2.0--inviter-0--invevery-10_2022_01_02_06_26_45_0000--s-0/"
            exp_specs["scale_env_with_demo_stats"] = False
            exp_specs["plan_step"] = 1000
            pkl_name = "/params.pkl"
        elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
            train_file = ""
            exp_specs["scale_env_with_demo_stats"] = False
            exp_specs["plan_step"] = 1000

        exp_specs["policy_checkpoint"] = "./logs/" + train_file + _file + pkl_name
        flag = False
        if "dpo" in train_file:
            flag = True

        seed = exp_specs["seed"]
        set_seed(seed)
        experiment(exp_specs, flag)
