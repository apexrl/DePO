import torch
import gym
import math
import itertools
import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

from gail_torch.policy import BasePolicy
from gail_torch.model import discrete_actor, discrete_critic, iresnet
from gail_torch.utils import estimate_advantages, normal_log_density

EPS = 0  # 1e-7


class DPO_W(BasePolicy):
    def __init__(
        self,
        expert_memory,
        id_lr=1e-3,
        id_batch_size=64,
        sp_batch_size=64,
        sp_alpha=1.0,
        sp_lr=1e-3,
        actor_lr=3e-4,
        critic_lr=3e-4,
        l2_reg=1e-3,
        gamma=0.95,
        eps_clip=0.1,
        gae_lambda=0.95,
        num_units=128,
        max_epochs_since_update=1,
        optim_epochs=10,
        mini_batch_size=64,
        valid_ratio=0.2,
        max_valid=200,
        max_num_inverse_dynamic_updates_per_loop_iter=0,
        num_blocks=10,
        sp_discriminator=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.observation_space, gym.spaces.Box)
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.obs_dim = self.observation_space.shape[0]
        self.act_num = self.action_space.n
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.optim_epochs = optim_epochs
        self.num_units = num_units

        self.sp_discriminator = sp_discriminator

        self.critic = discrete_critic(self.obs_dim, num_units)

        self.inverse_dynamic = nn.Sequential(
            nn.Linear(self.obs_dim * 2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, self.act_num),
        )
        # self.inverse_dynamic = iresnet(
        #     self.obs_dim * 2,
        #     num_blocks,
        #     self.act_num
        # )
        self.state_predictor = nn.Sequential(
            nn.Linear(self.obs_dim, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, self.obs_dim),
        )
        self.id_lr = id_lr
        self.id_optim = torch.optim.Adam(self.inverse_dynamic.parameters(), lr=id_lr)
        self.sp_optim = torch.optim.Adam(self.state_predictor.parameters(), lr=sp_lr)
        self.actor_optim = torch.optim.Adam(
            self.state_predictor.parameters(), lr=actor_lr
        )
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.sp_alpha = sp_alpha

        self.id_batch_size = id_batch_size
        self.sp_batch_size = sp_batch_size
        self.expert_memory = expert_memory
        self.random_action = False
        self.deterministic_action = False
        self.valid_ratio = valid_ratio
        self.max_valid = max_valid
        self.max_num_inverse_dynamic_updates_per_loop_iter = (
            max_num_inverse_dynamic_updates_per_loop_iter
        )

        self._policy_cnt = 0
        self._inv_cnt = 0
        self._sp_cnt = 0

    def reset_inverse_dynamics(self, num_units=None, obs_dim=None, act_num=None):
        if obs_dim is None:
            obs_dim = self.obs_dim
        if act_num is None:
            act_num = self.act_num
        if num_units is None:
            num_units = self.num_units
        self.inverse_dynamic = nn.Sequential(
            nn.Linear(obs_dim * 2, self.num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, act_num),
        )
        self.id_optim = torch.optim.Adam(
            self.inverse_dynamic.parameters(), lr=self.id_lr
        )

    def get_action(self, obs, deterministic=False):
        act_info = {}
        obs = torch.from_numpy(obs).to(self.device, torch.float)
        if self.random_action:
            act_logits = torch.ones(self.act_num).to(self.device).unsqueeze(0)
        else:
            next_obs_logits = self.state_predictor(obs.unsqueeze(0))
            # if self.deterministic_action or deterministic:
            #     next_obs_idx = next_obs_logits.argmax(dim=-1)[0]
            # else:
            #     next_obs_dist = Categorical(logits=next_obs_logits)
            #     next_obs_idx = next_obs_dist.sample()[0]
            # next_obs = torch.zeros(self.obs_dim).to(self.device)
            # next_obs[next_obs_idx] = 1
            gumbels = (
                -torch.empty_like(next_obs_logits).exponential_().log()
            )  # ~Gumbel(0,1)
            gumbels = (next_obs_logits + gumbels) / 1  # ~Gumbel(logits,tau)
            y_soft = gumbels.softmax(dim=-1)
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(next_obs_logits).scatter_(-1, index, 1.0)
            next_obs = (y_hard - y_soft.detach() + y_soft).squeeze(0)
            # next_obs = torch.softmax(next_obs_logits, axis=-1).squeeze(0)
            _input = torch.cat((obs, next_obs), dim=-1).unsqueeze(0)
            act_logits = self.inverse_dynamic(_input)
            act_info["next_obs_logits"] = next_obs_logits
        assert not torch.max(torch.isnan(act_logits)), "nan0, {}, {}".format(
            next_obs_logits, act_logits
        )

        if self.deterministic_action or deterministic:
            act = act_logits.argmax(dim=-1)[0]
        else:
            act_dist = Categorical(logits=act_logits)
            act = act_dist.sample()[0]
        act_info["act_logits"] = act_logits
        return act, act_info

    def get_log_prob(self, x, actions):
        obs = x
        next_obs_logits = self.state_predictor(x.unsqueeze(0))
        gumbels = (
            -torch.empty_like(next_obs_logits).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (next_obs_logits + gumbels) / 1  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=-1)
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(next_obs_logits).scatter_(-1, index, 1.0)
        next_obs = (y_hard - y_soft.detach() + y_soft).squeeze(0)
        # next_obs = torch.softmax(next_obs_logits, axis=-1).squeeze(0)
        _input = torch.cat((obs, next_obs), dim=-1).unsqueeze(0)
        act_logits = self.inverse_dynamic(_input)
        action_prob = torch.softmax(act_logits, dim=-1).squeeze(0)
        return torch.log(
            action_prob.gather(1, torch.where(actions > 0)[1].unsqueeze(1))
        )

    def get_sp_log_prob(self, obs, obs_prime):
        next_obs_logits = self.state_predictor(obs.unsqueeze(0))
        next_obs_prob = torch.softmax(next_obs_logits, dim=-1).squeeze(0) + EPS
        return torch.log(
            next_obs_prob.gather(1, torch.where(obs_prime > 0)[1].unsqueeze(1))
        )

    def get_sp_inv_log_prob(self, obs, obs_prime, actions):
        next_obs_logits = self.state_predictor(obs.unsqueeze(0))
        next_obs_prob = torch.softmax(next_obs_logits, dim=-1).squeeze(0)

        gumbels = (
            -torch.empty_like(next_obs_logits).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (next_obs_logits + gumbels) / 1  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim=-1)
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(next_obs_logits).scatter_(-1, index, 1.0)
        next_obs = (y_hard - y_soft.detach() + y_soft).squeeze(0)
        # next_obs = torch.softmax(next_obs_logits, axis=-1).squeeze(0)
        _input = torch.cat((obs, next_obs), dim=-1).unsqueeze(0)
        act_logits = self.inverse_dynamic(_input)
        action_prob = torch.softmax(act_logits, dim=-1).squeeze(0)

        return torch.log(
            next_obs_prob.gather(1, torch.where(obs_prime > 0)[1].unsqueeze(1))
        ) + torch.log(action_prob.gather(1, torch.where(actions > 0)[1].unsqueeze(1)))

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

    def update_sup(self, memory):

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

        self.id_optim.zero_grad()
        id_loss.backward()
        self.id_optim.step()

        expert_batch = self.expert_memory.sample(self.sp_batch_size)

        expert_obs = torch.from_numpy(expert_batch["observation"]).to(
            self.device, torch.float
        )
        expert_next_obs = torch.from_numpy(expert_batch["next_observation"]).to(
            self.device, torch.float
        )

        pred_next_obs_logits = self.state_predictor(expert_obs)

        sp_loss = F.nll_loss(
            F.log_softmax(
                pred_next_obs_logits.reshape(-1, pred_next_obs_logits.shape[-1]), dim=-1
            ),
            expert_next_obs.reshape(-1, expert_next_obs.shape[-1]).argmax(dim=-1),
        )

        self.sp_optim.zero_grad()
        sp_loss.backward()
        self.sp_optim.step()

        self._cnt += 1

        print(sp_loss, id_loss)

        return {"sp_loss": sp_loss.item(), "id_loss": id_loss.item()}

    def update_sp(self):

        expert_batch = self.expert_memory.sample(self.sp_batch_size)

        expert_obs = torch.from_numpy(expert_batch["observation"]).to(
            self.device, torch.float
        )
        expert_next_obs = torch.from_numpy(expert_batch["next_observation"]).to(
            self.device, torch.float
        )

        pred_next_obs_logits = self.state_predictor(expert_obs)

        sp_loss = F.nll_loss(
            F.log_softmax(
                pred_next_obs_logits.reshape(-1, pred_next_obs_logits.shape[-1]), dim=-1
            ),
            expert_next_obs.reshape(-1, expert_next_obs.shape[-1]).argmax(dim=-1),
        )

        self.sp_optim.zero_grad()
        sp_loss.backward()
        self.sp_optim.step()

        self._sp_cnt += 1

        return {"sp_loss": sp_loss.item()}

    def update_inv(self, memory):

        data_size = len(memory)  # get all data

        split_idx_sets = range(data_size)

        all_data = memory.sample(-1)

        # Split into training and valid sets
        num_valid = min(int(data_size * self.valid_ratio), self.max_valid)
        num_train = data_size - num_valid
        permutation = np.random.permutation(split_idx_sets)

        train_all_data = {}
        valid_all_data = {}
        for key in all_data:
            # train_all_data[key] = all_data[key][np.concatenate([permutation[num_valid:],unsplit_idx_sets]).astype(np.int32)]
            train_all_data[key] = all_data[key][permutation[num_valid:]]
            valid_all_data[key] = all_data[key][permutation[:num_valid]]

        print("[ Invdynamics ] Training {} | Valid: {}".format(num_train, num_valid))
        idxs = np.random.randint(num_train, size=num_train)

        if self.max_num_inverse_dynamic_updates_per_loop_iter:
            epoch_iter = range(self.max_num_inverse_dynamic_updates_per_loop_iter)
        else:
            epoch_iter = itertools.count()

        # epoch_iter = range(50)

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[idxs]

        grad_updates = 0
        batch_size = self.id_batch_size
        break_train = False
        self.best_valid = 1e-7
        self._epochs_since_update = 0

        for inv_train_epoch in epoch_iter:
            if break_train:
                break
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                obs = torch.from_numpy(train_all_data["observation"][batch_idxs]).to(
                    self.device, torch.float
                )
                act = torch.from_numpy(train_all_data["action"][batch_idxs]).to(
                    self.device, torch.float
                )
                next_obs = torch.from_numpy(
                    train_all_data["next_observation"][batch_idxs]
                ).to(self.device, torch.float)
                # print(obs.argmax(axis=-1), next_obs.argmax(axis=-1), act.argmax(axis=-1))

                _input = torch.cat([obs, next_obs], dim=-1)
                pred_act_logits = self.inverse_dynamic(_input)

                id_loss = F.nll_loss(
                    F.log_softmax(
                        pred_act_logits.reshape(-1, pred_act_logits.shape[-1]), dim=-1
                    ),
                    act.reshape(-1, act.shape[-1]).argmax(dim=-1),
                )

                self.id_optim.zero_grad()
                id_loss.backward()
                self.id_optim.step()

                self._inv_cnt += 1

                # return {"sp_loss": sp_loss.item(), "id_loss": id_loss.item()}

            idxs = shuffle_rows(idxs)

            ### Do validation
            if num_valid > 0:
                valid_obs = torch.from_numpy(valid_all_data["observation"]).to(
                    self.device, torch.float
                )
                valid_acts_logits = torch.from_numpy(valid_all_data["action"]).to(
                    self.device, torch.float
                )
                valid_acts = valid_acts_logits.argmax(dim=-1)[0]
                valid_next_obs = torch.from_numpy(
                    valid_all_data["next_observation"]
                ).to(self.device, torch.float)
                valid_pred_acts_logits = self.inverse_dynamic(
                    torch.cat([valid_obs, valid_next_obs], dim=-1)
                )
                valid_pred_acts = valid_pred_acts_logits.argmax(dim=-1)[0]
                valid_acc = (
                    (valid_acts == valid_pred_acts).type(torch.FloatTensor).mean()
                )

                break_train = self.valid_break(inv_train_epoch, valid_acc)

        return {"valid_acc": valid_acc.item()}

    def valid_break(self, train_epoch, valid_acc):
        updated = False
        current = valid_acc
        best = self.best_valid
        improvement = current - best
        if improvement == 0:
            return True
        if improvement > 0.005:
            self.best_valid = current
            updated = True
            improvement = current - best
            # print('epoch {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(train_epoch, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            print(
                "[ Invdynamics ] Breaking at epoch {}: {} epochs since update ({} max with valid acc)".format(
                    train_epoch,
                    self._epochs_since_update,
                    self._max_epochs_since_update,
                    valid_acc,
                )
            )
            return True
        else:
            return False

    def update_policy(self, memory):

        batch = memory.collect()

        obs = torch.from_numpy(batch["observation"]).to(self.device, torch.float)
        act = torch.from_numpy(batch["action"]).to(self.device, torch.int64)
        done = torch.tensor(batch["done"], dtype=torch.float, device=self.device)
        next_obs = torch.from_numpy(batch["next_observation"]).to(
            self.device, torch.float
        )

        if self.discriminator is not None:
            with torch.no_grad():
                _input = torch.cat([obs, next_obs], dim=-1)
                rew = self.discriminator.get_reward(_input)
        else:
            rew = torch.tensor(
                batch["reward"],
                dtype=torch.float,
                device=self.device,
            )

        with torch.no_grad():
            values = self.critic(obs)
            # fixed_log_probs = self.get_log_prob(obs, act)
            fixed_log_probs = self.get_sp_log_prob(obs, next_obs)
            # fixed_log_probs = self.get_sp_inv_log_prob(obs, next_obs, act)

        advantages, returns = estimate_advantages(
            rew, done, values, self.gamma, self.gae_lambda, self.device
        )

        policy_losses = []
        value_losses = []

        advantages = advantages - torch.min(advantages)

        optim_iter_num = int(math.ceil(obs.shape[0] / self.mini_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(obs.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            obs, act, next_obs, returns, advantages, fixed_log_probs = (
                obs[perm].clone(),
                act[perm].clone(),
                next_obs[perm].clone(),
                returns[perm].clone(),
                advantages[perm].clone(),
                fixed_log_probs[perm].clone(),
            )

            for i in range(optim_iter_num):
                expert_batch = self.expert_memory.sample(self.sp_batch_size)

                expert_obs = torch.from_numpy(expert_batch["observation"]).to(
                    self.device, torch.float
                )
                expert_next_obs = torch.from_numpy(expert_batch["next_observation"]).to(
                    self.device, torch.float
                )

                pred_next_obs_logits = self.state_predictor(expert_obs)

                sp_loss = F.nll_loss(
                    F.log_softmax(
                        pred_next_obs_logits.reshape(
                            -1, pred_next_obs_logits.shape[-1]
                        ),
                        dim=-1,
                    ),
                    expert_next_obs.reshape(-1, expert_next_obs.shape[-1]).argmax(
                        dim=-1
                    ),
                )

                ind = slice(
                    i * self.mini_batch_size,
                    min((i + 1) * self.mini_batch_size, obs.shape[0]),
                )
                obs_b, act_b, advantages_b, returns_b, fixed_log_probs_b, next_obs_b = (
                    obs[ind],
                    act[ind],
                    advantages[ind],
                    returns[ind],
                    fixed_log_probs[ind],
                    next_obs[ind],
                )

                # log_probs = self.get_log_prob(obs_b, act_b)
                log_probs = self.get_sp_log_prob(obs_b, next_obs_b)
                # log_probs = self.get_sp_inv_log_prob(obs_b, next_obs_b, act_b)
                # print("obs:", obs_b.argmax(axis=1))
                # print("next_obs:", next_obs_b.argmax(axis=1))
                # print("log_probs:", log_probs)
                # print("advantages:", advantages_b)

                values_pred = self.critic(obs_b)
                value_loss = (values_pred - returns_b).pow(2).mean()

                for param in self.critic.parameters():
                    value_loss += param.pow(2).sum() * self.l2_reg
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()
                value_losses.append(value_loss.item())

                # print(log_probs.shape, advantages_b.shape)
                ratios = torch.exp(log_probs - fixed_log_probs_b)
                surr1 = ratios * advantages_b
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * advantages_b
                )
                # policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss = (-1.0 * log_probs * advantages_b).mean()

                policy_loss = policy_loss + self.sp_alpha * sp_loss

                self.actor_optim.zero_grad()
                policy_loss.backward()
                a = torch.nn.utils.clip_grad_norm_(
                    self.state_predictor.parameters(), 10
                )
                # print(a)
                self.actor_optim.step()
                policy_losses.append(policy_loss.item())

        if self.writer is not None:
            prefix = f"Agent_{self.agent_id}/" if self.agent_id is not None else ""
            self.writer.add_scalar(
                prefix + "Loss/policy_loss", np.mean(policy_losses), self._cnt
            )
            self.writer.add_scalar(
                prefix + "Loss/value_loss", np.mean(value_losses), self._cnt
            )

        self._policy_cnt += 1

    def update(self, memory):
        return
