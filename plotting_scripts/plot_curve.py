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

colors = ["red", "royalblue", "purple", "darkorange", "green", "orchid", "darkcyan"]


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
    num_method = 4
    line_width = 2.5

    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.3)

    # InvertedPendulum
    env = "invpendulum"
    window_length = 10
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = "gail-lfo-invpendulum-union--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"
        elif i == 1:
            name = "sl-lfo-invpendulum-STANDARD-EXP--splr-0.01--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[0][0].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[0][0].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[0, 0].set_title("InvertedPendulum-v2", fontsize=title_size, pad=titel_pad)
    ax[0, 0].set_xlabel("steps", fontsize=lable_size)
    ax[0, 0].set_ylabel("Averaged return", fontsize=lable_size)
    ax[0, 0].set_xlim(left=0, right=20)
    ax[0, 0].set_ylim(bottom=-500)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6e6"]
    ax[0, 0].set_xticks(x_index)
    ax[0, 0].set_xticklabels(x_label, fontsize=ticksize)

    # InvertedDoublePendulum
    env = "invdoublependulum"
    window_length = 10
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = "gail-lfo-invdoublependulum-union--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"
        elif i == 1:
            name = "sl-lfo-invdoublependulum-STANDARD-EXP--splr-0.01--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[0][1].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[0][1].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[0, 1].set_title("InvertedDoublePendulum-v2", fontsize=title_size, pad=titel_pad)
    ax[0, 1].set_xlabel("steps", fontsize=lable_size)
    ax[0, 1].set_ylabel("Averaged return", fontsize=lable_size)
    ax[0, 1].set_xlim(left=0, right=20)
    ax[0, 1].set_ylim(bottom=-500)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
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
        "2.0e6",
    ]
    ax[0, 1].set_xticks(x_index)
    ax[0, 1].set_xticklabels(x_label, fontsize=ticksize)

    # Hopper
    env = "hopper"
    window_length = 10
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = (
                "gail-lfo-hopper-union--ms-2--gp-4.0--spalpha-1.0--idbeta-0.5--rs-2.0"
            )
        elif i == 1:
            name = "sl-lfo-hopper-STANDARD-EXP--splr-0.01--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[0][2].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[0][2].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[0, 2].set_title("Hopper-v2", fontsize=title_size, pad=titel_pad)
    ax[0, 2].set_xlabel("steps", fontsize=lable_size)
    ax[0, 2].set_ylabel("Averaged return", fontsize=lable_size)
    ax[0, 2].set_xlim(left=0, right=20)
    ax[0, 2].set_ylim(bottom=-500)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6e7"]
    ax[0, 2].set_xticks(x_index)
    ax[0, 2].set_xticklabels(x_label, fontsize=ticksize)

    # Walker
    env = "walker"
    window_length = 15
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = "gail-lfo-walker-union-test--cycle--gp-8.0--spalpha-1.0--idbeta-0.5--rs-2.0"
        elif i == 1:
            name = "sl-lfo-walker-STANDARD-EXP--splr-0.01--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[1][0].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[1][0].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[1, 0].set_title("Walker-v2", fontsize=title_size, pad=titel_pad)
    ax[1, 0].set_xlabel("steps", fontsize=lable_size)
    ax[1, 0].set_ylabel("Averaged return", fontsize=lable_size)
    ax[1, 0].set_xlim(left=0, right=20)
    ax[1, 0].set_ylim(bottom=-500)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
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
        "2.6e7",
    ]

    ax[1, 0].set_xticks(x_index)
    ax[1, 0].set_xticklabels(x_label, fontsize=ticksize)

    # Halfcheetah
    env = "halfcheetah"
    window_length = 15
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = "gail-lfo-halfcheetah-union--ms-1--gp-0.5--spalpha-0.35--idbeta-0.25--rs-2.0"
        elif i == 1:
            name = "sl-lfo-halfcheetah-STANDARD-EXP--splr-0.001--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[1][1].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[1][1].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[1, 1].set_title("HalfCheetah-v2", fontsize=title_size, pad=titel_pad)
    ax[1, 1].set_xlabel("steps", fontsize=lable_size)
    ax[1, 1].set_ylabel("Averaged return", fontsize=lable_size)
    ax[1, 1].set_xlim(left=0, right=20)
    ax[1, 1].set_ylim(bottom=-1000)
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
    ax[1, 1].set_xticks(x_index)
    ax[1, 1].set_xticklabels(x_label, fontsize=ticksize)

    # Ant
    env = "ant"
    window_length = 15
    require_suffix = None
    for i in range(num_method):
        name = names[i] + "-" + env + "-STANDARD-EXP"
        if i == 0:
            name = "gail-lfo-ant-union--ms-1--gp-0.5--spalpha-1.2--idbeta-0.5--rs-2.0"
        elif i == 1:
            name = "sl-lfo-ant-STANDARD-EXP--splr-0.001--idlr-0.0001"
        returns = read_multi_files("logs/" + name, require_suffix)
        returns = np.array(returns).T
        print("logs/" + name, returns.shape)
        returns = np.vstack([-100 * np.ones(returns.shape[1]), returns])
        x = np.arange(returns.shape[0]) + 1
        if ("gail-lfo" in names[i]) or ("sl-lfo" in names[i]) or ("bco" in names[i]):
            x = x.astype(np.float64) + 0.5
        y1 = r_mins(returns)
        y2 = r_maxs(returns)
        ma_y1 = MA(y1, window_length)
        ma_y2 = MA(y2, window_length)
        ax[1][2].fill_between(
            x,
            ma_y1,
            ma_y2,
            where=ma_y2 >= ma_y1,
            facecolor=colors[i],
            interpolate=True,
            alpha=alpha,
        )
        ax[1][2].plot(
            x,
            MA(r_means(returns), window_length),
            color=colors[i],
            linewidth=line_width,
            label=methods[i],
        )

    ax[1, 2].set_title("Ant-v2", fontsize=title_size, pad=titel_pad)
    ax[1, 2].set_xlabel("steps", fontsize=lable_size)
    ax[1, 2].set_ylabel("Averaged return", fontsize=lable_size)
    ax[1, 2].set_xlim(left=0, right=20)
    ax[1, 2].set_ylim(bottom=-500)
    x_index = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1,0", "1.2", "1.4", "1.6e7"]
    ax[1, 2].set_xticks(x_index)
    ax[1, 2].set_xticklabels(x_label, fontsize=ticksize)

    # ax[0, 0].grid()
    # ax[0, 1].grid()
    # ax[0, 2].grid()
    # ax[1, 0].grid()
    # ax[1, 1].grid()
    # ax[1, 2].grid()

    ax[1][1].legend(
        methods,
        ncol=num_method,
        loc="upper left",
        bbox_to_anchor=(-0.2, -0.15),
        fontsize="x-large",
        frameon=False,
    )

    # plt.tight_layout()
    plt.savefig("curves.pdf", bbox_inches="tight")
    plt.show()
