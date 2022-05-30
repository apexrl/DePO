import torch
import gym
import math
import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

from gail_torch.policy import BasePolicy
from gail_torch.model import discrete_actor, discrete_critic
from gail_torch.utils import estimate_advantages


class BCO(BasePolicy):
    def __init__(
        self,
        expert_memory,
        id_lr=1e-3,
        id_batch_size=64,
        actor_batch_size=64,
        actor_lr=1e-3,
        num_units=128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.observation_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.obs_dim = self.observation_space.shape[0]
        self.act_num = self.action_space.n
        self.inverse_dynamic = nn.Sequential(
            nn.Linear(self.obs_dim * 2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, self.act_num),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, self.act_num),
        )
        self.id_optim = torch.optim.Adam(self.inverse_dynamic.parameters(), lr=id_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.id_batch_size = id_batch_size
        self.actor_batch_size = actor_batch_size
        self.expert_memory = expert_memory
        self.random_action = False
        self.deterministic_action = False

    def get_action(self, obs):
        if self.random_action:
            act_logits = torch.ones(self.act_num).to(self.device).unsqueeze(0)
        else:
            act_logits = self.actor(
                torch.from_numpy(obs).to(self.device, torch.float).unsqueeze(0)
            )
        if self.deterministic_action:
            act = act_logits.argmax(dim=-1)[0]
        else:
            act_dist = Categorical(logits=act_logits)
            act = act_dist.sample()[0]
        act_info = {
            "act_logits": act_logits,
        }
        return act, act_info

    # def pretrain(self, memory):
    #     batch = memory.sample(self.id_batch_size)

    #     obs = torch.from_numpy(batch["observation"]).to(self.device, torch.float)
    #     act = torch.from_numpy(batch["action"]).to(self.device, torch.float)
    #     next_obs = torch.from_numpy(batch["next_observation"]).to(self.device, torch.float)

    #     _input = torch.cat([obs, next_obs], dim=-1)
    #     pred_act_logits = self.inverse_dynamic(_input)

    #     id_loss = F.nll_loss(
    #         F.log_softmax(pred_act_logits.reshape(-1, pred_act_logits.shape[-1]), dim=-1),
    #         act.reshape(-1, act.shape[-1]).argmax(dim=-1),
    #     )
    #     id_loss += 0.01 * pred_act_logits.pow(2).mean()

    #     self.id_optim.zero_grad()
    #     id_loss.backward()
    #     self.id_optim.step()

    def update(self, memory):

        batch = memory.sample(self.id_batch_size)

        obs = torch.from_numpy(batch["observation"]).to(self.device, torch.float)
        act = torch.from_numpy(batch["action"]).to(self.device, torch.float)
        next_obs = torch.from_numpy(batch["next_observation"]).to(
            self.device, torch.float
        )

        _input = torch.cat([obs, next_obs], dim=-1)
        pred_act_logits = self.inverse_dynamic(_input)

        id_loss = F.nll_loss(
            F.log_softmax(
                pred_act_logits.reshape(-1, pred_act_logits.shape[-1]), dim=-1
            ),
            act.reshape(-1, act.shape[-1]).argmax(dim=-1),
        )
        # id_loss += 0.01 * pred_act_logits.pow(2).mean()

        self.id_optim.zero_grad()
        id_loss.backward()
        self.id_optim.step()

        expert_batch = self.expert_memory.sample(self.actor_batch_size)

        expert_obs = torch.from_numpy(expert_batch["observation"]).to(
            self.device, torch.float
        )
        # expert_act = torch.from_numpy(expert_batch["action"]).to(self.device, torch.float)
        expert_next_obs = torch.from_numpy(expert_batch["next_observation"]).to(
            self.device, torch.float
        )

        _input = torch.cat([expert_obs, expert_next_obs], dim=-1)

        expert_act_logits = self.inverse_dynamic(_input)

        policy_act_logits = self.actor(expert_obs)

        actor_loss = F.nll_loss(
            F.log_softmax(
                policy_act_logits.reshape(-1, policy_act_logits.shape[-1]), dim=-1
            ),
            expert_act_logits.argmax(dim=-1),
        )
        # actor_loss += 0.01 * policy_act_logits.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._cnt += 1

        return {"actor_loss": actor_loss.item(), "id_loss": id_loss.item()}
