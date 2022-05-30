import os
import numpy as np
import matplotlib.pyplot as plt
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


methods = ["BCO", "GAIfO", "GAIfO-DP", "DPO (w/o PG)", "DPO (w PG)"]  #
names = ["bco", "gailfo", "gailfo-dp", "sl-lfo", "gail-lfo"]  #
envs = ["invpendulum", "invdoublependulum", "hopper", "walker", "halfcheetah", "ant"]

if __name__ == "__main__":

    num_method = 5

    for i in range(num_method):
        means = []
        stds = []
        for env in envs:
            file_path = "../final_performance/{}/{}/res.csv".format(names[i], env)
            mean, std = compute_res(file_path)
            means.append(mean)
            stds.append(std)

        print(
            "{} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {}\\\\".format(
                methods[i],
                means[0],
                stds[0],
                means[1],
                stds[1],
                means[2],
                stds[2],
                means[3],
                stds[3],
                means[4],
                stds[4],
                means[5],
                stds[5],
            )
        )
