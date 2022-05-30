from torch import nn
import torch
import gym
from abc import ABC
import torch.nn.functional as F

from gail_torch.model import discriminator_net


class Discriminator(ABC, nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        lr=3e-4,
        num_units=128,
        agent_id=None,
        state_only=False,
        device=torch.device("cpu"),
        expert_memory=None,
        writer=None,
    ):
        super(Discriminator, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.writer = writer
        self.device = device
        self.expert_memory = expert_memory
        self.state_only = state_only
        if state_only:
            num_inputs = self.observation_space.shape[0] * 2  # obs and next_obs
        else:
            if isinstance(self.action_space, gym.spaces.Box):
                num_inputs = (
                    self.observation_space.shape[0] + self.action_space.shape[0]
                )
            else:
                num_inputs = self.observation_space.shape[0] + self.action_space.n
                self._discrete_act = True
                self._act_num = self.action_space.n
        self.discriminator = discriminator_net(num_inputs, num_units)
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr
        )
        self._cnt = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["writer"]
        return state

    def update(self, memory, expert_memory=None):
        if expert_memory is None:
            assert self.expert_memory is not None
            expert_memory = self.expert_memory

        batch_size = len(memory)
        agent_batch = memory.collect()
        assert (
            len(expert_memory) > batch_size
        ), "expert dataset must be larger than batch size"
        expert_batch = expert_memory.sample(batch_size)

        agent_obs = torch.from_numpy(agent_batch["observation"]).to(
            self.device, torch.float
        )
        if self.state_only:
            agent_next_obs = torch.from_numpy(agent_batch["next_observation"]).to(
                self.device, torch.float
            )
            agent_input = torch.cat([agent_obs, agent_next_obs], dim=-1)
        else:
            agent_act = torch.from_numpy(agent_batch["action"]).to(
                self.device, torch.float
            )
            agent_input = torch.cat([agent_obs, agent_act], dim=-1)
        agent_prob = self.discriminator(agent_input)

        expert_obs = torch.from_numpy(expert_batch["observation"]).to(
            self.device, torch.float
        )
        if self.state_only:
            expert_next_obs = torch.from_numpy(expert_batch["next_observation"]).to(
                self.device, torch.float
            )
            expert_input = torch.cat([expert_obs, expert_next_obs], dim=1)
        else:
            expert_act = torch.from_numpy(expert_batch["action"]).to(
                self.device, torch.float
            )
            expert_input = torch.cat([expert_obs, expert_act], dim=1)
        expert_prob = self.discriminator(expert_input)

        self.discriminator_optim.zero_grad()
        loss = F.binary_cross_entropy(
            agent_prob, torch.zeros_like(agent_prob)
        ) + F.binary_cross_entropy(expert_prob, torch.ones_like(expert_prob))
        loss.backward()
        self.discriminator_optim.step()

        if self.writer is not None:
            self.writer.add_scalar("Loss/discriminator_loss", loss, self._cnt)

        self._cnt += 1
        return {"discriminator_loss": loss.item()}

    def get_reward(self, input_):
        input_ = input_.to(self.device)
        disc_prob = self.discriminator(input_).squeeze()
        disc_rew = torch.log(disc_prob)
        return disc_rew
