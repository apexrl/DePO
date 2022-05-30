import torch

from gail_torch.utils.memory import Memory
from gail_torch.utils.tools import (
    estimate_advantages,
    make_env,
    flat_dict_to_episode,
    generate_tune_config,
)
from gail_torch.utils.math import normal_log_density


def np_batch_to_torch(batch, device):
    keys = list(batch.keys())
    for k in keys:
        try:
            batch[k] = torch.from_numpy(batch[k]).to(device, torch.float)
        except TypeError:
            batch.pop(k)
    return batch


def torch_batch_to_np(batch):
    keys = list(batch.keys())
    for k in keys:
        try:
            batch[k] = batch[k].detach().cpu().numpy()
        except TypeError:
            batch.pop(k)
    return batch


def get_self_data(data, agent_id):
    # TODO(zbzhu): Fix this.
    if len(data.shape) == 2:
        return data[..., agent_id]
    return data[..., agent_id, :]


def get_oppo_data(data, agent_id):
    return torch.cat((data[..., :agent_id, :], data[..., agent_id + 1 :, :]), dim=1)
