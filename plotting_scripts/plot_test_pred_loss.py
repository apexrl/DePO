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

colors = [
    "red",
    "royalblue",
    "green",
    "purple",
    "darkorange",
    "mistyrose",
    "orchid",
    "darkcyan",
    "lightsalmon",
    "gold",
    "olive",
    "slateblue",
    "black",
    "darkblue",
    "deeppink",
    "slategray",
    "lime",
    "silver",
    "lightcyan",
    "darkkhaki",
    "teal",
    "palegoldenrod",
    "goldenrod",
    "lemonchiffon",
    "cadetblue",
]


def read_multi_files(folder_path, require_suffix=None, column="AverageReturn"):
    min_len = 400
    suffix = ""

    files = os.listdir(folder_path + suffix)
    reward_data = []
    tmp = []

    for file in files:
        if (require_suffix is not None) and (file[-6] not in require_suffix):
            continue
        file_path = folder_path + suffix + "/" + file + "/progress.csv"
        print("reading from " + file_path)
        df = pd.read_csv(file_path)
        min_len = min(min_len, len(df))
        tmp.append(df[column])

    for data in tmp:
        reward_data.append(data[:min_len])

    return reward_data


def MA(value, step):
    ma_value = []
    for i in range(len(value)):
        if i < 5:
            tmp = value[i : i + int(step / 1.5)]
        elif 5 <= i < 10:
            tmp = value[i : i + int(step / 1.3)]
        elif 10 <= i < 15:
            tmp = value[i : i + int(step / 1.1)]
        else:
            tmp = value[i : i + step]
        if len(tmp) > 0:
            ma_value.append(sum(tmp) / len(tmp))
    return ma_value


r_means = lambda x: np.nanmean(x, axis=1)
r_stderrs = lambda x: np.nanstd(x, axis=1) / np.sqrt(np.count_nonzero(x, axis=1))
r_mins = lambda x: r_means(x) - r_stderrs(x)  # np.nanmin(x, axis=1)
r_maxs = lambda x: r_means(x) + r_stderrs(x)  # np.nanmax(x, axis=1)

methods = ["DPO", "DPO (w.o. PG)", "GAILFO", "BCO"]  # , 'GAILFO-DP'
names = ["gail-lfo", "sl-lfo", "gailfo", "bco"]  # , 'gailfo-dp'

if __name__ == "__main__":
    title_size = 20
    lable_size = 18
    ticksize = 15
    line_width = 2.5
    num_ablation = 5

    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    fig, ax = plt.subplots(1, 4, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.3)

    methods = [
        "DPO w.o. Reg",
        "DPO w. M.S-1 Reg",
        "DPO w. M.S-2 Reg",
        "DPO w. Cycle Reg",
        "DPO w. Cycle-M.S-1 Reg",
    ]

    # hopper
    env = "hopper"
    window_length = 15
    require_suffix = [
        "-union--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union--ms-1--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union--ms-2--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union-test--cycle--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union-test--cycle--ms-1--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0",
    ]

    for i in range(len(require_suffix)):
        suffix = require_suffix[i]
        returns = read_multi_files(
            "logs/" + "gail-lfo" + "-" + env, suffix, "Test Pred_Real_MSE"
        )
        returns = np.array(returns).T
        returns = np.vstack([np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[0].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[0].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[0].set_title("Hopper-v2", fontsize=title_size, pad=titel_pad)
    ax[0].set_xlabel("steps", fontsize=lable_size)
    ax[0].set_ylabel("Test Pred Real MSE", fontsize=lable_size)
    ax[0].hlines(-100, 0, 20, colors="green", linestyles="dashed", linewidth=2.5)
    ax[0].set_xlim(left=0, right=20)
    ax[0].set_ylim(bottom=0)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6e7"]
    ax[0].set_xticks(x_index)
    ax[0].set_xticklabels(x_label, fontsize=ticksize)

    # Walker
    env = "walker"
    window_length = 15

    require_suffix = [
        "-union--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union--ms-1--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union--ms-2--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union-test--cycle--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0",
        "-union-test--cycle--ms-1--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0",
    ]

    for i in range(len(require_suffix)):
        suffix = require_suffix[i]
        returns = read_multi_files(
            "logs/" + "gail-lfo" + "-" + env, suffix, "Test Pred_Real_MSE"
        )
        returns = np.array(returns).T
        print(suffix, returns.shape)
        returns = np.vstack([np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[1].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[1].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[1].set_title("Walker-v2", fontsize=title_size, pad=titel_pad)
    ax[1].set_xlabel("steps", fontsize=lable_size)
    ax[1].set_ylabel("Test Pred Real MSE", fontsize=lable_size)
    ax[1].hlines(-100, 0, 20, colors="green", linestyles="dashed", linewidth=2.5)
    ax[1].set_xlim(left=0, right=20)
    ax[1].set_ylim(bottom=0)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    x_label = [
        "0",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "1",
        "1.2",
        "1.4",
        "1.6",
        "1.8",
        "2.0",
        "2.2",
        "2.4",
        "2.6e7",
    ]
    ax[1].set_xticks(x_index)

    # Half
    env = "halfcheetah"
    window_length = 15

    require_suffix_half = [
        "-union--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
        "-union--ms-1--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
        "-union--ms-2--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
        "-union-test--cycle--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
        "-union-test--cycle--ms-1--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0",
    ]

    for i in range(len(require_suffix_half)):
        suffix = require_suffix_half[i]
        returns = read_multi_files(
            "logs/" + "gail-lfo" + "-" + env, suffix, "Test Pred_Real_MSE"
        )
        returns = np.array(returns).T
        print(suffix, returns.shape)
        returns = np.vstack([np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[2].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[2].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[2].set_title("Half-v2", fontsize=title_size, pad=titel_pad)
    ax[2].set_xlabel("steps", fontsize=lable_size)
    ax[2].set_ylabel("Test Pred Real MSE", fontsize=lable_size)
    # ax[2].hlines(-100, 0, 20, colors='green', linestyles='dashed', linewidth=2.5)
    ax[2].set_xlim(left=0, right=20)
    ax[2].set_ylim(bottom=0)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    x_label = [
        "0",
        "0.2",
        "0.4",
        "0.6",
        "0.8",
        "1.0",
        "1.2",
        "1.4",
        "1.6",
        "1.8",
        "2.0",
        "2.2",
        "2.4",
        "2.6",
        "2.8",
        "3.0e7",
    ]
    ax[2].set_xticks(x_index)
    ax[2].set_xticklabels(x_label, fontsize=ticksize)

    # Ant
    env = "ant"
    window_length = 15

    require_suffix = [
        "-union--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
        "-union--ms-1--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
        "-union--ms-2--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
        "-union-test--cycle--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
        "-union-test--cycle--ms-1--gp-0.5--spalpha-1.1--idbeta-0.5--rs-2.0",
    ]

    for i in range(len(require_suffix)):
        suffix = require_suffix[i]
        returns = read_multi_files(
            "logs/" + "gail-lfo" + "-" + env, suffix, "Test Pred_Real_MSE"
        )
        returns = np.array(returns).T
        print(suffix, returns.shape)
        returns = np.vstack([np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[3].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[3].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[3].set_title("Ant-v2", fontsize=title_size, pad=titel_pad)
    ax[3].set_xlabel("steps", fontsize=lable_size)
    ax[3].set_ylabel("Test Pred Real MSE", fontsize=lable_size)
    # ax[3].hlines(-100, 0, 20, colors='green', linestyles='dashed', linewidth=2.5)
    ax[3].set_xlim(left=0, right=20)
    ax[3].set_ylim(bottom=0)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6e7"]
    ax[3].set_xticks(x_index)
    ax[3].set_xticklabels(x_label, fontsize=ticksize)

    ax[3].legend(
        methods,
        ncol=num_ablation,
        loc="upper left",
        bbox_to_anchor=(-1.0, -0.15),
        fontsize="x-large",
        frameon=False,
    )

    plt.savefig("test_pred.pdf", bbox_inches="tight")
    plt.show()
