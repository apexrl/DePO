import torch
import gym
import math
import numpy as np
from torch import nn
from torch.distributions import Categorical
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

from gail_torch.policy import BasePolicy
from gail_torch.model import continuous_actor, continuous_critic
from gail_torch.utils import estimate_advantages


class PPOPolicy(BasePolicy):
    def __init__(
        self,
        actor_lr=3e-4,
        critic_lr=3e-4,
        l2_reg=1e-3,
        gamma=0.95,
        eps_clip=0.1,
        gae_lambda=0.95,
        optim_epochs=10,
        mini_batch_size=64,
        num_units=128,
        activation="tanh",
        **kwargs: Any,
    ) -> None:
        super(PPOPolicy, self).__init__(**kwargs)
        assert isinstance(self.observation_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Box)
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]
        self.actor = continuous_actor(
            self.obs_dim, self.act_dim, num_units, activation=activation
        )
        self.critic = continuous_critic(self.obs_dim, num_units)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.optim_epochs = optim_epochs
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.l2_reg = l2_reg

    def get_action(self, obs):
        act_mean, _, act_std = self.actor(
            torch.from_numpy(obs).to(self.device, torch.float).unsqueeze(0)
        )
        act = torch.normal(act_mean, act_std)[0]
        return act, {}

    def update(self, memory):
        batch = memory.collect()

        obs = torch.from_numpy(batch["observation"]).to(self.device, torch.float)
        act = torch.from_numpy(batch["action"]).to(self.device, torch.int64)
        next_obs = torch.from_numpy(batch["next_observation"]).to(
            self.device, torch.float
        )
        done = torch.tensor(batch["done"], dtype=torch.float, device=self.device)

        if self.discriminator is not None:
            with torch.no_grad():
                if self.discriminator.state_only:
                    disc_input = torch.cat((obs, next_obs), dim=1)
                else:
                    disc_input = torch.cat((obs, act), dim=1)
                rew = self.discriminator.get_reward(disc_input)
        else:
            rew = torch.tensor(
                batch["reward"],
                dtype=torch.float,
                device=self.device,
            )

        with torch.no_grad():
            values = self.critic(obs)
            fixed_log_probs = self.actor.get_log_prob(obs, act)

        advantages, returns = estimate_advantages(
            rew, done, values, self.gamma, self.gae_lambda, self.device
        )

        actor_losses = []
        critic_losses = []

        optim_iter_num = int(math.ceil(obs.shape[0] / self.mini_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(obs.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            obs, act, returns, advantages, fixed_log_probs = (
                obs[perm].clone(),
                act[perm].clone(),
                returns[perm].clone(),
                advantages[perm].clone(),
                fixed_log_probs[perm].clone(),
            )

            for i in range(optim_iter_num):
                ind = slice(
                    i * self.mini_batch_size,
                    min((i + 1) * self.mini_batch_size, obs.shape[0]),
                )
                obs_b, act_b, advantages_b, returns_b, fixed_log_probs_b = (
                    obs[ind],
                    act[ind],
                    advantages[ind],
                    returns[ind],
                    fixed_log_probs[ind],
                )

                log_probs = self.actor.get_log_prob(obs_b, act_b)

                values_pred = self.critic(obs_b)
                value_loss = (values_pred - returns_b).pow(2).mean()

                for param in self.critic.parameters():
                    value_loss += param.pow(2).sum() * self.l2_reg
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()
                critic_losses.append(value_loss.item())

                ratios = torch.exp(log_probs - fixed_log_probs_b)
                surr1 = ratios * advantages_b
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * advantages_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                self.actor_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optim.step()
                actor_losses.append(policy_loss.item())

        if self.writer is not None:
            self.writer.add_scalar("Loss/actor_loss", np.mean(actor_losses), self._cnt)
            self.writer.add_scalar(
                "Loss/critic_loss", np.mean(critic_losses), self._cnt
            )

        self._cnt += 1

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
        }
