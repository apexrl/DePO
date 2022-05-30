from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle

algo_names = ["dpo", "gaifo", "bco"]
colors = ["red", "blue", "green"]
repeat_ratio = 1
window_length = 1


def MA(value, step):
    ma_value = []
    for i in range(len(value)):
        if step > 1:
            if i < 5:
                tmp = value[i : i + int(step / 1.5)]
            elif 5 <= i < 10:
                tmp = value[i : i + int(step / 1.3)]
            elif 10 <= i < 15:
                tmp = value[i : i + int(step / 1.1)]
            else:
                tmp = value[i : i + step]
        else:
            tmp = [value[i]]
        if len(tmp) > 0:
            ma_value.append(sum(tmp) / len(tmp))
        else:
            ma_value.append(tmp[0])
    return np.array(ma_value)


r_means = lambda x: np.nanmean(x, axis=1)
r_stderrs = lambda x: np.nanstd(x, axis=1) / np.sqrt(np.count_nonzero(x, axis=1))
r_mins = lambda x: r_means(x) - r_stderrs(x)  # np.nanmin(x, axis=1)
r_maxs = lambda x: r_means(x) + r_stderrs(x)  # np.nanmax(x, axis=1)

dpo_rews = []
bco_rews = []
gaifo_rews = []

for seed in [0, 1, 2]:

    try:
        with open(f"dpo_{repeat_ratio}_seed{seed}.pkl", "rb") as f:
            dpo_metrics = pickle.load(f)
        dpo_rews.append(dpo_metrics["avg_rew"])
    except:
        pass

    with open(f"bco_{repeat_ratio}_seed{seed}.pkl", "rb") as f:
        bco_metrics = pickle.load(f)

    with open(f"gaifo_{repeat_ratio}_seed{seed}.pkl", "rb") as f:
        gaifo_metrics = pickle.load(f)

    bco_rews.append(bco_metrics["avg_rew"])
    gaifo_rews.append(gaifo_metrics["avg_rew"])

dpo_rews = np.array(dpo_rews).T
bco_rews = np.array(bco_rews).T
gaifo_rews = np.array(gaifo_rews).T

fig = plt.figure(figsize=(10, 5))
plt.clf()

# key = "jsd"
# idx = 0
key = "avg_rew"
idx = 0

# plt.plot(np.arange(len(dpo_metrics[key])), dpo_metrics[key], color=colors[0])

# plt.plot(np.arange(len(bco_metrics[key])), bco_metrics[key], color=colors[1])

# plt.plot(np.arange(len(gaifo_metrics[key])), gaifo_metrics[key], color=colors[2])

x = np.arange(dpo_rews.shape[0])
x[11:] = x[11:] - 1
y1 = r_mins(dpo_rews)
y2 = r_maxs(dpo_rews)
ma_y1 = MA(y1, window_length)
ma_y2 = MA(y2, window_length)
plt.fill_between(
    x,
    ma_y1,
    ma_y2,
    where=ma_y2 >= ma_y1,
    facecolor=colors[0],
    interpolate=True,
    alpha=0.5,
)
plt.plot(x, MA(r_means(dpo_rews), window_length), color=colors[0], linewidth=2)

x = np.arange(bco_rews.shape[0])
x[11:] = x[11:] - 1
y1 = r_mins(bco_rews)
y2 = r_maxs(bco_rews)
ma_y1 = MA(y1, window_length)
ma_y2 = MA(y2, window_length)
plt.fill_between(
    x,
    ma_y1,
    ma_y2,
    where=ma_y2 >= ma_y1,
    facecolor=colors[1],
    interpolate=True,
    alpha=0.5,
)
plt.plot(x, MA(r_means(bco_rews), window_length), color=colors[1], linewidth=2)

x = np.arange(gaifo_rews.shape[0])
x[11:] = x[11:] - 1
y1 = r_mins(gaifo_rews)
y2 = r_maxs(gaifo_rews)
ma_y1 = MA(y1, window_length)
ma_y2 = MA(y2, window_length)
plt.fill_between(
    x,
    ma_y1,
    ma_y2,
    where=ma_y2 >= ma_y1,
    facecolor=colors[2],
    interpolate=True,
    alpha=0.5,
)
plt.plot(x, MA(r_means(gaifo_rews), window_length), color=colors[2], linewidth=2)


plt.ylim(0, 1.45)
patches = [
    # mpatches.Patch(color="red", label="Average Reward"),
    Line2D([0], [0], lw=2, label="DePO", color=colors[0]),
    Line2D([0], [0], lw=2, label="BCO", color=colors[1]),
    Line2D([0], [0], lw=2, label="GAIFO", color=colors[2]),
]
plt.legend(handles=patches, fontsize=14)
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Success Rate", fontsize=20)
x_index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
x_label = ["0", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200"]
# plt.gca().set_ylim(bottom=0, top=1.0)
plt.gca().set_xticks(x_index)
plt.gca().set_xticklabels(x_label, fontsize=12)
x_index = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
x_label = ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "120", "140", "160", "180", "200"]
plt.yticks(fontsize=15)

plt.text(4, 1.2, "k=1", fontsize=25)
plt.text(14, 1.2, "k=4", fontsize=25)

# plt.text(4, 1.05, "k=1", fontsize=25)
# plt.text(14, 1.05, "k=4", fontsize=25)

plt.vlines(10, 0, 2.5, colors="k", linestyles="dashed")

plt.savefig(f"metric_{repeat_ratio}.pdf", dpi=600, bbox_inches="tight")
