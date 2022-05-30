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

from video import save_video


def experiment(variant, lfo):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

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

    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    if lfo:
        state_predictor = joblib.load(variant["policy_checkpoint"])["state_predictor"]
        inverse_dynamic = joblib.load(variant["policy_checkpoint"])["inverse_dynamic"]

        policy = ReparamTanhMultivariateGaussianLfOPolicy(
            hidden_sizes=num_hidden * [net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_diff=variant["state_diff"],
            state_predictor=state_predictor,
            inverse_dynamic=inverse_dynamic,
        )
    else:
        policy = joblib.load(variant["policy_checkpoint"])["exploration_policy"][0]

    if variant["eval_deterministic"]:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    eval_sampler = PathSampler(
        env,
        policy,
        variant["num_eval_steps"],
        variant["max_path_length"],
        no_terminal=variant["no_terminal"],
        render=variant["render"],
        render_kwargs=variant["render_kwargs"],
        render_mode=variant["render_mode"],
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = eval_util.get_average_returns(test_paths)
    std_returns = eval_util.get_std_returns(test_paths)
    print(average_returns, std_returns)

    if variant["render"] and variant["render_mode"] == "rgb_array":
        video_path = variant["video_path"]
        video_path = os.path.join(video_path, variant["env_specs"]["env_name"])

        print("saving videos...")
        for i, test_path in enumerate(test_paths):
            images = np.stack(test_path["image"], axis=0)
            fps = 1 // getattr(env, "dt", 1 / 30)
            video_save_path = os.path.join(video_path, f"episode_{i}.mp4")
            save_video(images, video_save_path, fps=fps)

    return average_returns, std_returns, test_paths


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

    seed = exp_specs["seed"]
    set_seed(seed)
    # setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    train_file = (
        exp_specs["method"] + "-" + exp_specs["env_specs"]["env_name"] + "-STANDARD-EXP"
    )
    pkl_name = "/best.pkl"

    if "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
        pkl_name = "/params.pkl"

    if "invpendulum" in exp_specs["env_specs"]["env_name"]:
        pkl_name = "/params.pkl"

    if "gail-lfo" in exp_specs["method"]:
        if "hopper" in exp_specs["env_specs"]["env_name"]:
            train_file = (
                "gail-lfo-hopper-union--ms-2--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"
            )
        elif "walker" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-lfo-walker-union-test--cycle--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0"
        elif "halfcheetah" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-lfo-halfcheetah-union-test--cycle--ms-1--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0"
        elif "ant" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-lfo-ant-union--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0"
        elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
            train_file = (
                "gail-lfo-invpendulum-union--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"
            )
        elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "gail-lfo-invdoublependulum-gail-union--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"

    if "sl-lfo" in exp_specs["method"]:
        train_file = (
            "sl-lfo-"
            + exp_specs["env_specs"]["env_name"]
            + "-STANDARD-EXP--splr-0.01--idlr-0.0001"
        )
        if "halfcheetah" in exp_specs["env_specs"]["env_name"]:
            train_file = "sl-lfo-halfcheetah-STANDARD-EXP--splr-0.001--idlr-0.0001"
            pkl_name = "/params.pkl"
        elif "ant" in exp_specs["env_specs"]["env_name"]:
            train_file = "sl-lfo-ant-STANDARD-EXP--splr-0.001--idlr-0.0001"
            pkl_name = "/params.pkl"
        elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "sl-lfo-invpendulum-STANDARD-EXP--splr-0.001--idlr-0.0001"
            pkl_name = "/params.pkl"
        elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
            train_file = (
                "sl-lfo-invdoublependulum-STANDARD-EXP--splr-0.001--idlr-0.0001"
            )
            pkl_name = "/params.pkl"

    if "bco" in exp_specs["method"]:
        if "halfcheetah" in exp_specs["env_specs"]["env_name"]:
            pkl_name = "/params.pkl"
        elif "ant" in exp_specs["env_specs"]["env_name"]:
            pkl_name = "/params.pkl"

    if "gailfo" in exp_specs["method"]:
        if "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-invdoublependulum-gail-STANDARD-EXP"

    if "gailfo-dp" in exp_specs["method"]:
        if "hopper" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-dp-hopper-STANDARD-EXP--gp-4.0--splr-0.01--idlr-0.001--rs-2.0--decay-1.0"
        elif "walker" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-dp-walker-STANDARD-EXP--bkup"
        elif "halfcheetah" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-dp-halfcheetah-STANDARD-EXP--gp-0.5--splr-0--idlr-0--rs-2.0--decay-1.0"
        elif "ant" in exp_specs["env_specs"]["env_name"]:
            train_file = (
                "gailfo-dp-ant-STANDARD-EXP--gp-0.5--splr-0--idlr-0--rs-2.0--decay-1.0"
            )
        elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-dp-invpendulum-10000-STANDARD-EXP--gp-4.0--splr-0.01--idlr-0.001--rs-2.0--decay-1.0"
        elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
            train_file = "gailfo-dp-invdoublependulum-gail-10000-STANDARD-EXP--gp-4.0--splr-0.01--idlr-0.001--rs-2.0--decay-1.0"

    train_files = [train_file]
    save_path = "./final_performance/"

    if exp_specs["ablation_study"]:
        save_path = "./ablation/"
        if "gail-lfo" in exp_specs["method"]:
            if "halfcheetah" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-halfcheetah-union--gp-0.5--spalpha-0.3--idbeta-0.25--rs-2.0",
                    "gail-lfo-halfcheetah-union--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
                    "gail-lfo-halfcheetah-union--gp-0.5--spalpha-0.4--idbeta-0.25--rs-2.0",
                    "gail-lfo-halfcheetah-union--gp-0.5--spalpha-0.45--idbeta-0.25--rs-2.0",
                    "gail-lfo-halfcheetah-union--gp-0.5--spalpha-0.5--idbeta-0.25--rs-2.0",
                ]
            elif "ant" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-ant-union--gp-0.5--spalpha-0.9--idbeta-0.5--rs-2.0",
                    "gail-lfo-ant-union--gp-0.5--spalpha-1.0--idbeta-0.5--rs-2.0",
                    "gail-lfo-ant-union--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
                    "gail-lfo-ant-union--gp-0.5--spalpha-1.2--idbeta-0.5--rs-2.0",
                    "gail-lfo-ant-union--gp-0.5--spalpha-1.3--idbeta-0.5--rs-2.0",
                ]

        if "gail-lfo-no-sp" in exp_specs["method"]:
            if "halfcheetah" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-halfcheetah-union-no-sp--gp-0.5--spalpha-0.0--idbeta-0.25--rs-2.0"
                ]
            elif "ant" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-ant-union-no-sp--gp-0.5--spalpha-0.0--idbeta-0.5--rs-2.0"
                ]
            elif "walker" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-walker-union-no-sp--gp-8.0--spalpha-0.0--idbeta-0.5--rs-2.0"
                ]
            elif "hopper" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-hopper-union-no-sp--gp-4.0--spalpha-0.0--idbeta-0.5--rs-2.0"
                ]
            elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-invpendulum-no-sp--gp-4.0--spalpha-0--idbeta-0.1--rs-2.0"
                ]
            elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
                train_files = [
                    "gail-lfo-invdoublependulum-union-no-sp--gp-4.0--spalpha-0--idbeta-0.05--rs-2.0"
                ]

        if "sl-lfo-consist" in exp_specs["method"]:
            if "halfcheetah" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-halfcheetah-zs--splr-0.001--idlr-0.0001"]
            elif "ant" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-ant-zs--splr-0.001--idlr-0.0001"]
            elif "walker" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-walker-zs--splr-0.01--idlr-0.0001"]
            elif "hopper" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-hopper-zs--splr-0.01--idlr-0.0001"]
            elif "invpendulum" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-invpendulum-zs--splr-0.001--idlr-0.0001"]
            elif "invdoublependulum" in exp_specs["env_specs"]["env_name"]:
                train_files = ["sl-lfo-invdoublependulum-zs--splr-0.001--idlr-0.0001"]

    for train_file in train_files:
        res_files = os.listdir("./logs/" + train_file)
        test_paths_all = []
        for file_ in res_files:
            exp_specs["policy_checkpoint"] = (
                "./logs/" + train_file + "/" + file_ + pkl_name
            )
            flag = False
            if "_lfo" in file_:
                flag = True
            average_returns, std_returns, test_paths = experiment(exp_specs, flag)
            test_paths_all.extend(test_paths)

            if args.save_res:
                save_dir = Path(save_path + train_file)
                save_dir.mkdir(exist_ok=True, parents=True)
                file_dir = save_dir.joinpath(
                    exp_specs["method"], exp_specs["env_specs"]["env_name"]
                )
                file_dir.mkdir(exist_ok=True, parents=True)

                if not os.path.exists(file_dir.joinpath("res.csv")):
                    with open(
                        save_dir.joinpath(
                            exp_specs["method"],
                            exp_specs["env_specs"]["env_name"],
                            "res.csv",
                        ),
                        "w",
                    ) as f:
                        f.write("avg,std\n")
                with open(
                    save_dir.joinpath(
                        exp_specs["method"],
                        exp_specs["env_specs"]["env_name"],
                        "res.csv",
                    ),
                    "a",
                ) as f:
                    f.write("{},{}\n".format(average_returns, std_returns))
        if exp_specs["save_samples"]:
            with open(
                Path(save_path).joinpath(
                    exp_specs["method"],
                    exp_specs["env_specs"]["env_name"],
                    "samples.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(test_paths_all, f)
