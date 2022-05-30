import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder

from tqdm import tqdm


class BCO(TorchBaseAlgorithm):
    """
    Depending on choice of reward function and size of replay
    buffer this will be:
        - AIRL
        - GAIL (without extra entropy term)
        - FAIRL
        - Discriminator Actor Critic

    I did not implement the reward-wrapping mentioned in
    https://arxiv.org/pdf/1809.02925.pdf though

    Features removed from v1.0:
        - gradient clipping
        - target disc (exponential moving average disc)
        - target policy (exponential moving average policy)
        - disc input noise
    """

    def __init__(
        self,
        mode,  # MLE or MSE
        inverse_mode,  # MLE or MSE
        inverse_dynamics,
        expert_replay_buffer,
        state_only=True,
        policy_optim_batch_size=1024,
        num_update_loops_per_train_call=1,
        num_policy_updates_per_loop_iter=1,
        num_inverse_dynamic_updates_per_loop_iter=1,
        num_pretrain_updates=10,
        pretrain_steps_per_epoch=5000,
        inverse_dynamic_lr=1e-3,
        inverse_dynamic_momentum=0.0,
        inverse_dynamic_optimizer_class=optim.Adam,
        policy_lr=1e-3,
        policy_momentum=0.0,
        policy_optimizer_class=optim.Adam,
        **kwargs
    ):
        assert mode in ["MSE", "MLE"], "Invalid bco algorithm!"
        assert inverse_mode in ["MSE", "MLE"], "Invalid bco algorithm!"
        if kwargs["wrap_absorbing"]:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.inverse_mode = inverse_mode
        self.state_only = state_only

        self.expert_replay_buffer = expert_replay_buffer

        self.policy_optim_batch_size = policy_optim_batch_size

        self.inverse_dynamics = inverse_dynamics
        self.inverse_dynamic_optimizer = inverse_dynamic_optimizer_class(
            inverse_dynamics.parameters(),
            lr=inverse_dynamic_lr,
            betas=(inverse_dynamic_momentum, 0.999),
        )

        self.policy_optimizer = policy_optimizer_class(
            self.exploration_policy.parameters(),
            lr=policy_lr,
            betas=(policy_momentum, 0.999),
        )

        self.inverse_dynamic_optim_batch_size = policy_optim_batch_size

        self.pretrain_steps_per_epoch = pretrain_steps_per_epoch

        print("\n\nINVERSE-DYNAMIC MOMENTUM: %f\n\n" % inverse_dynamic_momentum)

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.num_inverse_dynamic_updates_per_loop_iter = (
            num_inverse_dynamic_updates_per_loop_iter
        )
        self.num_pretrain_updates = num_pretrain_updates

        self.policy_eval_statistics = None

    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()

        if self.policy_eval_statistics is not None:
            self.eval_statistics.update(self.policy_eval_statistics)

        super().evaluate(epoch)

    def pretrain(self, *args, **kwargs):
        """
        Do anything before the main training phase.
        """
        print("Pretraining ...")
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for _ in tqdm(range(self.num_pretrain_updates)):
            # sample data using a random policy
            for steps_this_epoch in range(self.pretrain_steps_per_epoch):
                _, action, agent_info = self._get_action_and_info(observation)
                if self.render:
                    self.training_env.render()

                next_ob, raw_reward, terminal, env_info = self.training_env.step(action)
                if self.no_terminal:
                    terminal = False
                self._n_env_steps_total += 1

                reward = np.array([raw_reward])
                terminal = np.array([terminal])

                timeout = False
                if len(self._current_path_builder) >= (self.max_path_length - 1):
                    timeout = True
                timeout = np.array([timeout])

                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    np.array([False]) if self.no_terminal else terminal,
                    timeout,
                    absorbing=np.array([0.0, 0.0]),
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal[0]:
                    if self.wrap_absorbing:
                        raise NotImplementedError()
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                elif len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob
            self._do_inverse_dynamic_training(-1, False)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)
            for _ in range(self.num_inverse_dynamic_updates_per_loop_iter):
                self._do_inverse_dynamic_training(epoch, False)

    def _do_inverse_dynamic_training(self, epoch, use_expert_buffer=False):
        """
        Train the inverse dynamic model
        """
        self.inverse_dynamic_optimizer.zero_grad()

        batch = self.get_batch(
            self.inverse_dynamic_optim_batch_size,
            keys=["observations", "actions", "next_observations"],
            from_expert=use_expert_buffer,
        )

        obs = batch["observations"]
        acts = batch["actions"]
        next_obs = batch["next_observations"]

        if self.inverse_mode == "MLE":
            log_prob = self.inverse_dynamics.get_log_prob(obs, next_obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics[
                "Inverse-Dynamic-Log-Likelihood"
            ] = ptu.get_numpy(-1.0 * loss)
        else:
            pred_acts = self.inverse_dynamics(obs, next_obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics["Inverse-Dynamic-MSE"] = ptu.get_numpy(loss)

        assert not torch.max(
            torch.isnan(loss)
        ), "nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_std: {}".format(
            obs, obs_prime, acts, log_prob
        )
        loss.backward()

        self.inverse_dynamic_optimizer.step()

    def _do_policy_training(self, epoch, use_expert_buffer=True):
        batch = self.get_batch(
            self.policy_optim_batch_size,
            keys=["observations", "actions", "next_observations"],
            from_expert=use_expert_buffer,
        )

        obs = batch["observations"]
        obs_prime = batch["next_observations"]
        acts = self.inverse_dynamics(obs, obs_prime)[0]

        self.policy_optimizer.zero_grad()
        if self.mode == "MLE":
            log_prob = self.exploration_policy.get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics["Log-Likelihood"] = ptu.get_numpy(-1.0 * loss)
        else:
            pred_acts = self.exploration_policy(obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics["MSE"] = ptu.get_numpy(loss)
        loss.backward()
        self.policy_optimizer.step()

    @property
    def networks(self):
        return [self.inverse_dynamics, self.exploration_policy]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(inverse_dynamics=self.inverse_dynamics)
        return snapshot
