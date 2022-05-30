import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle

algo_names = ["dpo", "gaifo", "bco"]
colors = ["red", "blue", "green"]
repeat_ratio = 1


with open(f"dpo_{repeat_ratio}.pkl", "rb") as f:
    dpo_metrics = pickle.load(f)

with open(f"bco_{repeat_ratio}.pkl", "rb") as f:
    bco_metrics = pickle.load(f)

with open(f"gaifo_{repeat_ratio}.pkl", "rb") as f:
    gaifo_metrics = pickle.load(f)

# JSD
key = ""
fig = plt.figure(figsize=(5, 5))
plt.clf()
for idx, key in enumerate(dpo_metrics.keys()):
    plt.plot(np.arange(len(dpo_metrics[key])), dpo_metrics[key], color=colors[idx])

for idx, key in enumerate(bco_metrics.keys()):
    plt.plot(
        np.arange(len(bco_metrics[key])),
        bco_metrics[key],
        linestyle="--",
        color=colors[idx],
    )

plt.ylim(0, 5)
patches = [
    mpatches.Patch(color="red", label="JS Divergence"),
    mpatches.Patch(color="blue", label="Policy / SP Loss"),
    mpatches.Patch(color="green", label="ID Loss"),
    Line2D([0], [0], color="black", lw=2, label="DPO"),
    Line2D([0], [0], color="black", lw=2, label="BCO", ls="--"),
]
plt.legend(handles=patches)
plt.xlabel("Epoch")
plt.savefig(f"metric_{repeat_ratio}.pdf", dpi=600, bbox_inches="tight")


fig = plt.figure(figsize=(5, 5))
plt.clf()
for idx, key in enumerate(dpo_metrics.keys()):
    plt.plot(np.arange(len(dpo_metrics[key])), dpo_metrics[key], color=colors[idx])

for idx, key in enumerate(bco_metrics.keys()):
    plt.plot(
        np.arange(len(bco_metrics[key])),
        bco_metrics[key],
        linestyle="--",
        color=colors[idx],
    )

plt.ylim(0, 5)
patches = [
    mpatches.Patch(color="red", label="JS Divergence"),
    mpatches.Patch(color="blue", label="Policy / SP Loss"),
    mpatches.Patch(color="green", label="ID Loss"),
    Line2D([0], [0], color="black", lw=2, label="DPO"),
    Line2D([0], [0], color="black", lw=2, label="BCO", ls="--"),
]
plt.legend(handles=patches)
plt.xlabel("Epoch")
plt.savefig(f"metric_{repeat_ratio}.pdf", dpi=600, bbox_inches="tight")
