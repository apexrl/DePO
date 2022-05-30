import numpy as np
import seaborn
import pandas as pd

# seaborn.set(style = 'darkgrid')
seaborn.set_style("whitegrid")
title_size = 20
lable_size = 18
titel_pad = 10
alpha = 0.2


def compute_res(file_path, column="avg"):
    df = pd.read_csv(file_path)
    data = df[column]

    return np.round(data.mean(), 2), np.round(data.std(), 2)


methods = ["DPO (w/o SP)"]
names = ["gail-lfo-no-sp"]
# methods = ["DPO (w/o PG, w CL)"]
# names = ["sl-lfo-consist"]
envs = [
    "invpendulum",
    "invdoublependulum",
    "hopper",
    "walker",
    "halfcheetah",
    "ant",
]

if __name__ == "__main__":

    num_method = 5

    for env in envs:
        print(" & {}".format(env), end="")
    print("\\\\")

    for method, name in zip(methods, names):

        print(method, end="")

        for env in envs:
            file_path = "../ablation/{}/{}/res.csv".format(name, env)
            mean, std = compute_res(file_path)
            print(" & {} $\pm$ {}".format(mean, std), end="")
        print("\\\\")
