import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle

algo_names = ["dpo", "gaifo", "bco"]
colors = ["red", "blue", "green"]
repeat_ratio = 1


def data_to_img(data, img_shape=None):
    data = data / data.max()
    img_shape = [360, 360, 3]
    img = np.zeros(img_shape, dtype=np.float32)
    gs0 = int(img.shape[0] / data.shape[0])
    gs1 = int(img.shape[1] / data.shape[1])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1] = np.array(
                [1 - data[i, j], 1 - data[i, j], 1 - data[i, j]]
            )
    return img


with open(f"dpo_{repeat_ratio}.pkl", "rb") as f:
    dpo_metrics = pickle.load(f)

with open(f"bco_{repeat_ratio}.pkl", "rb") as f:
    bco_metrics = pickle.load(f)

with open(f"gaifo_{repeat_ratio}.pkl", "rb") as f:
    gaifo_metrics = pickle.load(f)

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

expert_heatmap = data_to_img(dpo_metrics["expert_density"][-1].reshape(6, 6))
axes[0].imshow(expert_heatmap)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("Expert")
# fig.savefig("expert_heatmap.png", dpi=300)

# agent_density = np.sum(dpo_metrics["agent_density"], axis=0)
# agent_heatmap = data_to_img(agent_density.reshape(6, 6))
agent_heatmap = data_to_img(dpo_metrics["agent_density"][-1].reshape(6, 6))
axes[1].imshow(agent_heatmap)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title("DPO")
# fig.savefig("dpo_heatmap.png", dpi=300)

# agent_density = np.sum(bco_metrics["agent_density"], axis=0)
# agent_heatmap = data_to_img(agent_density.reshape(6, 6))
agent_heatmap = data_to_img(bco_metrics["agent_density"][-1].reshape(6, 6))
axes[2].imshow(agent_heatmap)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_title("BCO")
# fig.savefig("bco_heatmap.png", dpi=300)

agent_density = np.sum(gaifo_metrics["agent_density"], axis=0)
agent_heatmap = data_to_img(agent_density.reshape(6, 6))
axes[3].imshow(agent_heatmap)
axes[3].set_xticks([])
axes[3].set_yticks([])
axes[3].set_title("GAIfO")
# fig.savefig("gaifo_heatmap.png", dpi=300)

plt.subplots_adjust(wspace=-0.3)
fig.savefig(f"heatmap_{repeat_ratio}.pdf", dpi=300, bbox_inches="tight")

bco_metrics.pop("agent_density")
dpo_metrics.pop("agent_density")
bco_metrics.pop("expert_density")
dpo_metrics.pop("expert_density")


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
