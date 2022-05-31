import numpy as np
from collections import OrderedDict
import itertools
import random
import gtimer as gt
import os

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder

from tqdm import tqdm


class AdvIRL_LfO(TorchBaseAlgorithm):
    """
    Main implementation of DePO (CDePG).
    """

    def __init__(
        self,
        mode,  # reward / train mode, see below
        inverse_mode,  # MLE or MSE
        state_predictor_mode,  # MLE or MSE
        discriminator,
        policy_trainer,
        expert_replay_buffer,
        state_only=False,
        sas=False,
        qss=False,
        sss=False,
        state_diff=False,
        union=False,
        union_sp=False,
        multi_step=False,
        reward_penelty=False,
        inv_buffer=False,
        update_weight=False,
        penelty_weight=1.0,
        step_num=1,
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,
        num_train_calls_between_inverse_dynamic_training=1,
        num_state_predictor_updates_per_loop_iter=100,
        max_num_inverse_dynamic_updates_per_loop_iter=None,
        num_inverse_dynamic_updates_per_loop_iter=0,
        num_pretrain_updates=0,
        pretrain_steps_per_epoch=5000,
        disc_lr=1e-3,
        disc_momentum=0.9,
        disc_optimizer_class=optim.Adam,
        state_predictor_lr=1e-3,
        state_predictor_alpha=20,
        state_predictor_momentum=0.0,
        state_predictor_optimizer_class=optim.Adam,
        inverse_dynamic_lr=1e-3,
        inverse_dynamic_beta=0.0,
        inverse_dynamic_momentum=0.0,
        inverse_dynamic_optimizer_class=optim.Adam,
        decay_ratio=1.0,
        use_grad_pen=True,
        use_wgan=True,
        grad_pen_weight=10,
        rew_clip_min=-10,
        rew_clip_max=10,
        valid_ratio=0.2,
        max_valid=5000,
        max_epochs_since_update=5,
        epsilon=0.0,
        min_epsilon=0.0,
        inv_buf_size=1000000,
        epsilon_ratio=1.0,
        rew_shaping=False,
        use_ensemble=False,
        pretrain_inv_num=50,
        changing_dynamics=False,
        num_train_epoch=None,
        **kwargs,
    ):
        assert mode in [
            "rl",
            "airl",
            "gail",
            "fairl",
            "gail2",
            "sl",
            "sl-test",
        ], "Invalid adversarial irl algorithm!"
        assert inverse_mode in ["MSE", "MLE", "MAE"], "Invalid bco algorithm!"
        # if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.inverse_mode = inverse_mode
        self.state_predictor_mode = state_predictor_mode
        self.state_only = state_only
        self.sas = sas
        self.qss = qss
        self.sss = sss
        self.state_diff = state_diff
        self.union = union
        self.union_sp = union_sp
        self.reward_penelty = reward_penelty
        self.penelty_weight = penelty_weight
        self.inv_buffer = inv_buffer
        self.update_weight = update_weight
        self.multi_step = multi_step
        self.step_num = step_num
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.num_inverse_dynamic_updates_per_loop_iter = (
            num_inverse_dynamic_updates_per_loop_iter
        )
        self.epsilon_ratio = epsilon_ratio
        self.rew_shaping = rew_shaping
        self.use_ensemble = use_ensemble
        self.pretrain_inv_num = pretrain_inv_num
        self.changing_dynamics = changing_dynamics
        self.num_train_epoch = num_train_epoch

        if epsilon > 0:
            print("\n EPSILON GREEDY! {}, RATIO {}".format(epsilon, epsilon_ratio))

        print("\n INV BUF SIZE {}!".format(inv_buf_size))

        print("\n PRE TRAIN NUM {}!".format(pretrain_inv_num))

        # For inv dynamics training's validation
        self.valid_ratio = valid_ratio
        self.max_valid = max_valid
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0

        self.num_train_calls_between_inverse_dynamic_training = (
            num_train_calls_between_inverse_dynamic_training
        )

        if self.mode in ["sl", "sl-test"]:
            self.union = False
            self.union_sp = False

        if self.union_sp:  # gail only train state predictor
            self.inverse_dynamic_beta = 0
            assert mode in ["rl", "airl", "gail", "fairl", "gail2"]

        self.expert_replay_buffer = expert_replay_buffer
        self.inv_replay_buffer = self.replay_buffer
        if self.inv_buffer:
            self.inv_replay_buffer = EnvReplayBuffer(
                inv_buf_size, self.env, random_seed=np.random.randint(10000)
            )
        self.target_state_predictor = None
        if self.multi_step:
            self.target_state_predictor = self.exploration_policy.state_predictor.copy()

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(), lr=disc_lr, betas=(disc_momentum, 0.999)
        )
        self.state_predictor_optimizer = state_predictor_optimizer_class(
            self.exploration_policy.state_predictor.parameters(),
            lr=state_predictor_lr,
            betas=(state_predictor_momentum, 0.999),
        )
        self.inverse_dynamic_optimizer = inverse_dynamic_optimizer_class(
            self.exploration_policy.inverse_dynamic.parameters(),
            lr=inverse_dynamic_lr,
            betas=(inverse_dynamic_momentum, 0.999),
        )
        self.state_predictor_alpha = state_predictor_alpha
        self.inverse_dynamic_beta = inverse_dynamic_beta

        # self.inverse_dynamic_scheduler = optim.lr_scheduler.ExponentialLR(
        #     self.inverse_dynamic_optimizer, gamma=decay_ratio
        # )

        # self.state_predictor_scheduler = optim.lr_scheduler.ExponentialLR(
        #     self.state_predictor_optimizer, gamma=decay_ratio
        # )

        self.decay_ratio = decay_ratio

        self.disc_optim_batch_size = disc_optim_batch_size
        self.state_predictor_optim_batch_size = policy_optim_batch_size
        self.inverse_dynamic_optim_batch_size = policy_optim_batch_size

        self.pretrain_steps_per_epoch = pretrain_steps_per_epoch

        print("\n\nDISC MOMENTUM: %f\n\n" % disc_momentum)
        print("\n\nSTATE-PREDICTOR MOMENTUM: %f\n\n" % state_predictor_momentum)
        print("\n\nINVERSE-DYNAMIC MOMENTUM: %f\n\n" % inverse_dynamic_momentum)
        if self.update_weight:
            print("\n\nUPDATE WEIGHT!\n\n")
        if self.union_sp:
            print("\n\nUNION STATE PREDICTOR!\n\n")
        if self.union:
            print("\n\nUNION TRAINING!\n\n")
        if self.reward_penelty:
            print("\n\nREWARD PENELTY!\n\n")
        if self.multi_step:
            print("\n\nMULTI STEP - {}!\n\n".format(self.step_num))
        if self.rew_shaping:
            print("\n\nREW SHAPING!\n\n")
        if self.use_ensemble:
            print("\n\nENSEMBLE INVERSE!\n\n")

        print(
            f"\n\nMax num_inverse_dynamic_updates_per_loop_iter: {max_num_inverse_dynamic_updates_per_loop_iter}\n\n"
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.use_wgan = use_wgan
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        if self.mode == "rl":
            self.num_disc_updates_per_loop_iter = 0
            self.state_predictor_alpha = 0.0
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.num_state_predictor_updates_per_loop_iter = (
            num_state_predictor_updates_per_loop_iter
        )
        # self.num_inverse_dynamic_updates_per_loop_iter = (
        #     num_inverse_dynamic_updates_per_loop_iter
        # )
        self.max_num_inverse_dynamic_updates_per_loop_iter = (
            max_num_inverse_dynamic_updates_per_loop_iter
        )
        self.num_pretrain_updates = num_pretrain_updates

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None
        self.policy_eval_statistics = None

    def get_batch(
        self,
        batch_size,
        from_expert,
        from_inv=False,
        keys=None,
        multi_step=False,
        step_num=1,
    ):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            if from_inv:
                buffer = self.inv_replay_buffer
            else:
                buffer = self.replay_buffer

        batch = buffer.random_batch(
            batch_size, keys=keys, multi_step=multi_step, step_num=step_num
        )
        batch = np_to_pytorch_batch(batch)
        return batch

    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        self.policy_eval_statistics = None
        # self.state_predictor_scheduler.step()
        # self.inverse_dynamic_scheduler.step()
        # self.epsilon *= max(self.min_epsilon, self.epsilon_ratio)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_ratio)

        if self.update_weight:
            self.state_predictor_alpha *= self.decay_ratio
            # self.inverse_dynamic_beta *= 1.1
            # self.state_predictor_alpha = min(10.0, self.state_predictor_alpha)
            # self.inverse_dynamic_beta = min(10.0, self.inverse_dynamic_beta)
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()

        if self.policy_eval_statistics is not None:
            self.eval_statistics.update(self.policy_eval_statistics)
        if "sl" not in self.mode:
            if self.disc_eval_statistics is not None:
                self.eval_statistics.update(self.disc_eval_statistics)
            policy_eval_statistics = self.policy_trainer.get_eval_statistics()
            if policy_eval_statistics is not None:
                self.eval_statistics.update()

        super().evaluate(epoch, pred_obs=True)

    def pretrain(self, pred_obs=False):
        """
        Do anything before the main training phase.
        """
        print("Pretraining ...")
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        pred_obs_prime = None

        for _ in tqdm(range(self.num_pretrain_updates)):
            # sample data using a random policy

            for steps_this_epoch in range(self.pretrain_steps_per_epoch):
                pred_obs_prime, action, agent_info = self._get_action_and_info(
                    observation, pred_obs
                )
                if self.render:
                    self.training_env.render()

                step_action = action.copy()
                if self.changing_dynamics:
                    step_action = -step_action
                next_ob, raw_reward, terminal, env_info = self.training_env.step(
                    step_action
                )
                act_log_prob = (np.array([0.0]),)
                if self.epsilon == 0:
                    obs = torch.Tensor([observation]).to(ptu.device)
                    act_log_prob = (
                        self.exploration_policy.get_log_prob(obs, action)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                add_inv_buf = terminal
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
                    np.array([False])
                    if (self.no_terminal or self.wrap_absorbing)
                    else terminal,
                    timeout,
                    pred_obs=pred_obs_prime,
                    absorbing=np.array([0.0, 0.0]),
                    agent_info=agent_info,
                    env_info=env_info,
                    action_log_prob=act_log_prob,
                    # add_inv_buf=add_inv_buf,
                )
                if terminal[0]:
                    if self.wrap_absorbing:
                        # raise NotImplementedError()
                        """
                        If we wrap absorbing states, two additional
                        transitions must be added: (s_T, s_abs) and
                        (s_abs, s_abs). In Disc Actor Critic paper
                        they make s_abs be a vector of 0s with last
                        dim set to 1. Here we are going to add the following:
                        ([next_ob,0], random_action, [next_ob, 1]) and
                        ([next_ob,1], random_action, [next_ob, 1])
                        This way we can handle varying types of terminal states.
                        """
                        # next_ob is the absorbing state
                        # for now just taking the previous action
                        self._handle_step(
                            next_ob,
                            # action,
                            self.training_env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            # next_ob,
                            np.zeros_like(next_ob),
                            np.array([False]),
                            timeout,
                            pred_obs=pred_obs_prime,
                            absorbing=np.array([0.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info,
                        )
                        self._handle_step(
                            # next_ob,
                            np.zeros_like(next_ob),
                            # action,
                            self.training_env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            # next_ob,
                            np.zeros_like(next_ob),
                            np.array([False]),
                            timeout,
                            pred_obs=pred_obs_prime,
                            absorbing=np.array([1.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info,
                        )
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                elif len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob
            # self._do_state_predictor_training(-1, True)
            self._do_inverse_dynamic_training(
                -1, False, valid_ratio=0.0, max_num=self.pretrain_inv_num
            )

    def _get_action_and_info(self, observation, pred_obs=False):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        pred_obs_prime = None
        agent_info = {}
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            if pred_obs:
                pred_obs_prime, action, agent_info = self.exploration_policy.get_action(
                    obs_np=observation, return_predicting_obs=pred_obs
                )
            else:
                action, agent_info = self.exploration_policy.get_action(
                    obs_np=observation
                )

        return pred_obs_prime, action, agent_info

    def _do_training(self, epoch):
        if self.num_train_epoch is not None:
            if self.num_train_epoch <= epoch:
                return
        # print(self._n_train_steps_totasl % self.num_train_calls_between_inverse_dynamic_training == 0)
        if (
            self._n_train_steps_total
            % self.num_train_calls_between_inverse_dynamic_training
            == 0
        ):
            if not self.union:
                if self.num_inverse_dynamic_updates_per_loop_iter > 0:
                    for _ in range(self.num_inverse_dynamic_updates_per_loop_iter):
                        self._do_batch_inverse_dynamic_training(epoch, False)
                else:
                    # train inverse dynamics until converged
                    self._do_inverse_dynamic_training(epoch, False)

        for t in range(self.num_update_loops_per_train_call):
            if not self.union:
                if not self.union_sp:  # union sp do not train state predictor
                    for _ in range(self.num_state_predictor_updates_per_loop_iter):
                        self._do_state_predictor_training(epoch, True)
            if "sl" not in self.mode:
                for _ in range(self.num_disc_updates_per_loop_iter):
                    self._do_reward_training(epoch)
                for _ in range(self.num_policy_updates_per_loop_iter):
                    self._do_policy_training(epoch)

    def _do_state_predictor_training(self, epoch, use_expert_buffer=True):
        """
        Train the state predictor
        """
        if self.mode != "sl":
            raise ValueError("should not be trained")
        self.state_predictor_optimizer.zero_grad()

        exp_keys = ["observations", "next_observations"]
        for i in np.arange(1, self.step_num + 1):
            exp_keys.append("next{}_observations".format(i))
        exp_batch = self.get_batch(
            self.state_predictor_optim_batch_size,
            keys=exp_keys,
            from_expert=use_expert_buffer,
            multi_step=self.multi_step,
            step_num=self.step_num,
        )

        agent_batch = self.get_batch(
            self.state_predictor_optim_batch_size,
            keys=["observations", "next_observations"],
            from_expert=False,
            multi_step=self.multi_step,
        )

        obs = exp_batch["observations"]
        next_obs = exp_batch["next_observations"]

        agent_obs = agent_batch["observations"]
        agent_next_obs = agent_batch["next_observations"]

        if self.state_predictor_mode == "MSE":
            pred_obs = self.exploration_policy.state_predictor(obs)[0]
            agent_pred_obs = self.exploration_policy.state_predictor(agent_obs)[0]

            label_obs = next_obs
            if self.state_diff:
                label_obs = next_obs - obs
            squared_diff = (pred_obs - label_obs) ** 2
            loss = torch.sum(squared_diff, dim=-1)
            if self.multi_step:
                next_pred_obs = pred_obs
                for i in np.arange(1, self.step_num + 1):
                    pred_obs_use = next_pred_obs
                    next_i_obs = exp_batch["next{}_observations".format(i)]
                    next_pred_obs = self.target_state_predictor(pred_obs_use)
                    squared_diff_2 = (next_pred_obs - next_i_obs) ** 2
                    loss += torch.sum(squared_diff_2, dim=-1)
                # loss = loss / (self.step_num + 1)
            loss = loss.mean()
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics["State-Pred-Expt-MSE"] = ptu.get_numpy(loss)

            agent_label_obs = agent_next_obs
            if self.state_diff:
                agent_label_obs = agent_next_obs - agent_obs
            agent_squared_diff = (agent_pred_obs - agent_label_obs) ** 2
            agent_loss = torch.sum(agent_squared_diff, dim=-1).mean()
            self.policy_eval_statistics["State-Pred-Real-MSE"] = ptu.get_numpy(
                agent_loss
            )
        elif self.inverse_mode == "MLE":
            log_prob = self.exploration_policy.state_predictor.get_log_prob(
                obs, next_obs
            )
            loss = -1.0 * log_prob.mean()
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics[
                "State-Predictor-Log-Likelihood"
            ] = ptu.get_numpy(-1.0 * loss)

        loss.backward()
        self.state_predictor_optimizer.step()

        if self.multi_step:
            ptu.copy_model_params_from_to(
                self.exploration_policy.state_predictor, self.target_state_predictor
            )

    def _do_batch_inverse_dynamic_training(self, epoch, use_expert_buffer=False):
        """
        Train the inverse dynamic model
        """
        self.inverse_dynamic_optimizer.zero_grad()

        batch = self.get_batch(
            self.inverse_dynamic_optim_batch_size,
            keys=["observations", "actions", "next_observations"],
            from_expert=use_expert_buffer,
            from_inv=True,
        )

        obs = batch["observations"]
        acts = batch["actions"]
        next_obs = batch["next_observations"]

        if self.inverse_mode == "MLE":
            log_prob = self.exploration_policy.inverse_dynamic.get_log_prob(
                obs, next_obs, acts
            )
            loss = -1.0 * log_prob
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics[
                "Inverse-Dynamic-Log-Likelihood"
            ] = ptu.get_numpy(-1.0 * loss.mean())

            assert not torch.max(
                torch.isnan(loss)
            ), "nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_std: {}".format(
                obs, next_obs, acts, log_prob
            )

        elif self.inverse_mode == "MSE":
            pred_obs = self.exploration_policy.inverse_dynamic(obs, next_obs)[0]
            squared_diff = (pred_obs - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1)
            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics["Inverse-Dynamic-MSE"] = ptu.get_numpy(
                loss.mean()
            )

        # # add inverse entropy for sac learning
        # policy_outputs = self.policy_trainer.policy(obs, return_log_prob=True)
        # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # ent_loss = self.policy_trainer.alpha * log_pi
        # self.policy_eval_statistics["Inv-Ent-Loss"] = ptu.get_numpy(ent_loss.mean())
        # loss = torch.mean(loss + ent_loss)
        loss = torch.mean(loss)

        loss.backward()
        # if torch.max(torch.isnan(torch.Tensor(list(self.exploration_policy.inverse_dynamic.parameters())))):
        # print("nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_std: {}".format(obs,obs_prime,acts,log_prob))
        # for name, parms in self.exploration_policy.inverse_dynamic.named_parameters():
        # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

        self.inverse_dynamic_optimizer.step()

    def _do_inverse_dynamic_training(
        self, epoch, use_expert_buffer=False, valid_ratio=None, max_num=None
    ):
        """
        Train the inverse dynamic model
        """
        if valid_ratio is None:
            valid_ratio = self.valid_ratio
        if max_num is None:
            max_num = self.max_num_inverse_dynamic_updates_per_loop_iter
        # Use the ensemble module to train inverse dynamics
        if self.use_ensemble:
            all_data = self.inv_replay_buffer.get_all(
                keys=["observations", "actions", "next_observations"],
            )
            train_inputs = np.concatenate(
                [all_data["observations"], all_data["next_observations"]], axis=-1
            )
            train_outputs = all_data["actions"]
            mse_loss = self.exploration_policy.inverse_dynamic.do_train(
                train_inputs, train_outputs
            )

            if self.policy_eval_statistics is None:
                self.policy_eval_statistics = OrderedDict()
            self.policy_eval_statistics["Inverse-Dynamic-MSE"] = mse_loss

            return

        data_size = self.inv_replay_buffer._size  # get all data

        current_policy_buf_size = self.replay_buffer._size
        split_idx_sets = range(data_size)
        unsplit_idx_sets = []

        all_data = self.inv_replay_buffer.get_all(
            keys=["observations", "actions", "next_observations"],
        )
        all_data = np_to_pytorch_batch(all_data)

        # if data_size > 2 * current_policy_buf_size:
        #     # make sure the newest data is always trained
        #     data_size = data_size - current_policy_buf_size
        #     policy_buf_start_idx = (self.inv_replay_buffer._top - current_policy_buf_size) % self.inv_replay_buffer._size
        #     policy_buf_end_idx = self.inv_replay_buffer._top
        #     assert policy_buf_start_idx != policy_buf_end_idx, (self.inv_replay_buffer._top, current_policy_buf_size, policy_buf_end_idx, policy_buf_start_idx, self.inv_replay_buffer._size)

        #     if policy_buf_start_idx > policy_buf_end_idx:
        #         split_idx_sets = list(range(policy_buf_end_idx,policy_buf_start_idx))
        #         unsplit_idx_sets = list(range(policy_buf_end_idx)) + list(range(policy_buf_start_idx,self.inv_replay_buffer._size))
        #     else:
        #         split_idx_sets = list(range(policy_buf_start_idx)) + list(range(policy_buf_end_idx,self.inv_replay_buffer._size))
        #         unsplit_idx_sets = list(range(policy_buf_start_idx,policy_buf_end_idx))

        # Split into training and valid sets
        num_valid = min(int(data_size * valid_ratio), self.max_valid)
        num_train = data_size - num_valid
        permutation = np.random.permutation(split_idx_sets)

        train_all_data = {}
        valid_all_data = {}
        for key in all_data:
            # train_all_data[key] = all_data[key][np.concatenate([permutation[num_valid:],unsplit_idx_sets]).astype(np.int32)]
            train_all_data[key] = all_data[key][permutation[num_valid:]]
            valid_all_data[key] = all_data[key][permutation[:num_valid]]

        print("[ Invdynamics ] Training {} | Valid: {}".format(num_train, num_valid))
        idxs = np.arange(num_train)

        if max_num:
            epoch_iter = range(max_num)
        else:
            epoch_iter = itertools.count()

        # epoch_iter = range(50)

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[idxs]

        grad_updates = 0
        batch_size = self.inverse_dynamic_optim_batch_size
        break_train = False
        self.best_valid = 10e7
        self._epochs_since_update = 0

        for inv_train_epoch in epoch_iter:
            idxs = shuffle_rows(idxs)
            if break_train:
                break
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                obs = train_all_data["observations"][batch_idxs]
                acts = train_all_data["actions"][batch_idxs]
                next_obs = train_all_data["next_observations"][batch_idxs]

                if self.inverse_mode == "MLE":
                    log_prob = self.exploration_policy.inverse_dynamic.get_log_prob(
                        obs, next_obs, acts
                    )
                    loss = -1.0 * log_prob
                    if self.policy_eval_statistics is None:
                        self.policy_eval_statistics = OrderedDict()
                    self.policy_eval_statistics[
                        "Inverse-Dynamic-Log-Likelihood"
                    ] = ptu.get_numpy(-1.0 * loss.mean())

                    assert not torch.max(
                        torch.isnan(loss)
                    ), "nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_prob: {}".format(
                        obs, next_obs, acts, log_prob
                    )

                elif self.inverse_mode == "MSE":
                    pred_acts = self.exploration_policy.inverse_dynamic(
                        obs, next_obs, deterministic=True
                    )[0]
                    squared_diff = (pred_acts - acts) ** 2
                    loss = torch.sum(squared_diff, dim=-1)
                    if self.policy_eval_statistics is None:
                        self.policy_eval_statistics = OrderedDict()
                    self.policy_eval_statistics["Inverse-Dynamic-MSE"] = ptu.get_numpy(
                        loss.mean()
                    )

                # # add inverse entropy for sac learning
                # policy_outputs = self.policy_trainer.policy(obs, return_log_prob=True)
                # new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
                # ent_loss = self.policy_trainer.alpha * log_pi
                # self.policy_eval_statistics["Inv-Ent-Loss"] = ptu.get_numpy(ent_loss.mean())
                # loss = torch.mean(loss + ent_loss)
                loss = torch.mean(loss)

                self.inverse_dynamic_optimizer.zero_grad()

                loss.backward()
                # if torch.max(torch.isnan(torch.Tensor(list(self.exploration_policy.inverse_dynamic.parameters())))):
                # print("nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_std: {}".format(obs,obs_prime,acts,log_prob))
                # for name, parms in self.exploration_policy.inverse_dynamic.named_parameters():
                # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

                self.inverse_dynamic_optimizer.step()

                pred_acts = self.exploration_policy.inverse_dynamic(
                    obs, next_obs, deterministic=True
                )[0]
                squared_diff = (pred_acts - acts) ** 2
                mse_loss = torch.sum(squared_diff, dim=-1)
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics["Inverse-Dynamic-MSE"] = ptu.get_numpy(
                    mse_loss.mean()
                )

            ### Do validation
            if num_valid > 0:
                valid_obs = valid_all_data["observations"]
                valid_acts = valid_all_data["actions"]
                valid_next_obs = valid_all_data["next_observations"]
                valid_pred_acts = self.exploration_policy.inverse_dynamic(
                    valid_obs, valid_next_obs, deterministic=True
                )[0]
                valid_squared_diff = (valid_pred_acts - valid_acts) ** 2
                valid_loss = torch.sum(valid_squared_diff, dim=-1).mean()
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics["Valid-InvDyn-MSE"] = ptu.get_numpy(
                    valid_loss
                )
                # print(ptu.get_numpy(loss), ptu.get_numpy(valid_loss))
                # print('[ Invdynamics ] {}, {}'.format(ptu.get_numpy(-loss), ptu.get_numpy(valid_loss)))

                break_train = self.valid_break(
                    inv_train_epoch, ptu.get_numpy(valid_loss)
                )

    def valid_break(self, train_epoch, valid_loss):
        updated = False
        current = valid_loss
        best = self.best_valid
        improvement = (best - current) / best
        # print(current, improvement)
        if improvement > 0.01:
            self.best_valid = current
            updated = True
            improvement = (best - current) / best
            # print('epoch {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(train_epoch, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            print(
                "[ Invdynamics ] Breaking at epoch {}: {} epochs since update ({} max)".format(
                    train_epoch,
                    self._epochs_since_update,
                    self._max_epochs_since_update,
                )
            )
            return True
        else:
            return False

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        self.disc_optimizer.zero_grad()

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        if self.sas:
            keys.append("next_observations")
            keys.append("actions")
        elif self.sss:
            keys.append("pred_observations")
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        expert_batch = self.get_batch(
            self.disc_optim_batch_size, from_expert=True, keys=keys
        )
        policy_batch = self.get_batch(
            self.disc_optim_batch_size, from_expert=False, keys=keys
        )

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            # pass
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        expert_next_obs = expert_batch["next_observations"]
        policy_next_obs = policy_batch["next_observations"]

        if self.wrap_absorbing:
            # pass
            expert_next_obs = torch.cat(
                [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
            )
            policy_next_obs = torch.cat(
                [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
            )

        expert_inputs = [expert_obs, expert_next_obs]
        policy_inputs = [policy_obs, policy_next_obs]

        if self.sas:
            expert_acts = self.exploration_policy.inverse_dynamic(
                expert_obs, expert_next_obs
            )[0]
            policy_acts = policy_batch["actions"]

            expert_inputs = [expert_obs, expert_acts, expert_next_obs]
            policy_inputs = [policy_obs, policy_acts, policy_next_obs]
        if self.sss:
            expert_pred_obs = expert_batch["next_observations"]
            policy_pred_obs = policy_batch["pred_observations"]

            expert_inputs = [expert_obs, expert_pred_obs, expert_next_obs]
            policy_inputs = [policy_obs, policy_pred_obs, policy_next_obs]

        expert_disc_input = torch.cat(expert_inputs, dim=1)
        policy_disc_input = torch.cat(policy_inputs, dim=1)

        if self.use_wgan:
            expert_logits = self.discriminator(expert_disc_input)
            policy_logits = self.discriminator(policy_disc_input)

            disc_ce_loss = -torch.sum(expert_logits) + torch.sum(policy_logits)
        else:
            disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

            disc_logits = self.discriminator(disc_input)
            disc_preds = (disc_logits > 0).type(disc_logits.data.type())
            disc_ce_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        assert not torch.max(
            torch.isnan(disc_total_loss)
        ), "nan-reward-training, disc_ce_loss: {}, disc_grad_pen_loss: {}".format(
            disc_ce_loss, disc_grad_pen_loss
        )
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            if not self.use_wgan:
                self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))
            if self.use_wgan:
                self.disc_eval_statistics["Expert D Logits"] = np.mean(
                    ptu.get_numpy(expert_logits)
                )
                self.disc_eval_statistics["Policy D Logits"] = np.mean(
                    ptu.get_numpy(policy_logits)
                )
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    def _do_policy_training(self, epoch):
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                from_expert=False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, from_expert=True
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(
                self.policy_optim_batch_size,
                from_expert=False,
                multi_step=self.multi_step,
            )

        if self.mode != "rl":
            obs = policy_batch["observations"]
            next_obs = policy_batch["next_observations"]

            if self.wrap_absorbing:
                # pass
                obs = torch.cat([obs, policy_batch["absorbing"][:, 0:1]], dim=-1)
                next_obs = torch.cat(
                    [next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )

            policy_inputs = [obs, next_obs]

            if self.sas:
                acts = policy_batch["actions"]
                policy_inputs = [obs, acts, next_obs]
            if self.sss:
                pred_next_obs = policy_batch["pred_observations"]
                policy_inputs = [obs, pred_next_obs, next_obs]

            self.discriminator.eval()
            disc_input = torch.cat(policy_inputs, dim=1)
            disc_logits = self.discriminator(disc_input).detach()
            self.discriminator.train()

            # compute the reward using the algorithm
            if self.mode == "airl":
                # If you compute log(D) - log(1-D) then you just get the logits
                policy_batch["rewards"] = disc_logits
            elif self.mode == "gail":
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=1
                )  # F.softplus(disc_logits, beta=-1)
            elif self.mode == "gail2":
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=-1
                )  # F.softplus(disc_logits, beta=-1)
            else:  # fairl
                policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

            self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Disc Rew Std"] = np.std(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Disc Rew Max"] = np.max(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Disc Rew Min"] = np.min(
                ptu.get_numpy(policy_batch["rewards"])
            )

            if self.reward_penelty:
                agent_pred_obs = self.exploration_policy.state_predictor(obs)
                pred_mse = (agent_pred_obs - next_obs) ** 2
                pred_mse = torch.sum(pred_mse, axis=-1, keepdim=True)
                reward_penelty = self.penelty_weight * pred_mse
                policy_batch["rewards"] -= reward_penelty

                self.disc_eval_statistics["Penelty Rew Mean"] = np.mean(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics["Penelty Rew Std"] = np.std(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics["Penelty Rew Max"] = np.max(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics["Penelty Rew Min"] = np.min(
                    ptu.get_numpy(reward_penelty)
                )

            if self.clip_max_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], max=self.rew_clip_max
                )
            if self.clip_min_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], min=self.rew_clip_min
                )
                if self.rew_shaping:
                    policy_batch["rewards"] -= self.rew_clip_min

        # policy optimization step
        if self.union or self.union_sp:
            exp_keys = ["observations", "next_observations", "actions"]
            for i in np.arange(1, self.step_num + 1):
                exp_keys.append("next{}_observations".format(i))

            expert_batch = self.get_batch(
                self.state_predictor_optim_batch_size,
                keys=exp_keys,
                from_expert=True,
                multi_step=self.multi_step,
                step_num=self.step_num,
            )
            inv_batch = None
            if self.inv_buffer and self.union:
                inv_batch = self.get_batch(
                    self.state_predictor_optim_batch_size,
                    keys=["observations", "next_observations", "actions"],
                    from_expert=False,
                    from_inv=True,
                )
            self.policy_trainer.train_step(
                policy_batch,
                qss=self.qss,
                alpha=self.state_predictor_alpha,
                beta=self.inverse_dynamic_beta,
                expert_batch=expert_batch,
                inv_batch=inv_batch,
                policy_optim_batch_size_from_expert=self.policy_optim_batch_size_from_expert,
                state_diff=self.state_diff,
                multi_step=self.multi_step,
                step_num=self.step_num,
                target_state_predictor=self.target_state_predictor,
            )
            if self.multi_step:
                ptu.copy_model_params_from_to(
                    self.exploration_policy.state_predictor, self.target_state_predictor
                )
        else:
            self.policy_trainer.train_step(policy_batch, qss=self.qss)

        if self.mode != "rl":
            self.disc_eval_statistics["Total Rew Mean"] = np.mean(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Total Rew Std"] = np.std(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Total Rew Max"] = np.max(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics["Total Rew Min"] = np.min(
                ptu.get_numpy(policy_batch["rewards"])
            )

    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        timeout,
        pred_obs,
        absorbing,
        agent_info,
        env_info,
        action_log_prob=None,
        add_inv_buf=True,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            pred_observations=pred_obs,
            absorbing=absorbing,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            timeout=timeout,
            pred_observations=pred_obs,
            absorbing=absorbing,
            agent_info=agent_info,
            env_info=env_info,
            action_log_prob=action_log_prob,
        )
        if self.inv_buffer and add_inv_buf:
            self.inv_replay_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_observation,
                timeout=timeout,
                absorbing=absorbing,
                agent_info=agent_info,
                env_info=env_info,
            )

    @property
    def networks(self):
        res = (
            [self.discriminator]
            + self.policy_trainer.networks
            + [
                self.policy_trainer.policy.state_predictor,
                self.policy_trainer.policy.inverse_dynamic,
            ]
        )
        if self.multi_step:
            res += [self.target_state_predictor]
        return res

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)

    def start_training(self, start_epoch=0, pred_obs=False):
        self.eval_statistics = None
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        pred_obs_prime = None

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            print("Training of epoch {}".format(epoch))
            for steps_this_epoch in tqdm(range(self.num_env_steps_per_epoch)):
                pred_obs_prime, action, agent_info = self._get_action_and_info(
                    observation, pred_obs
                )
                if self.render:
                    self.training_env.render()

                step_action = action.copy()
                if self.changing_dynamics:
                    step_action = -step_action
                next_ob, raw_reward, terminal, env_info = self.training_env.step(
                    step_action
                )
                act_log_prob = (np.array([0.0]),)
                if self.epsilon == 0:
                    obs = torch.Tensor([observation]).to(ptu.device)
                    act_log_prob = (
                        self.exploration_policy.get_log_prob(obs, action)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                add_inv_buf = terminal
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
                    np.array([False])
                    if (self.no_terminal or self.wrap_absorbing)
                    else terminal,
                    timeout,
                    pred_obs=pred_obs_prime,
                    absorbing=np.array([0.0, 0.0]),
                    agent_info=agent_info,
                    env_info=env_info,
                    action_log_prob=act_log_prob,
                    # add_inv_buf=add_inv_buf,
                )
                if terminal[0]:
                    if self.wrap_absorbing:
                        # raise NotImplementedError()
                        """
                        If we wrap absorbing states, two additional
                        transitions must be added: (s_T, s_abs) and
                        (s_abs, s_abs). In Disc Actor Critic paper
                        they make s_abs be a vector of 0s with last
                        dim set to 1. Here we are going to add the following:
                        ([next_ob,0], random_action, [next_ob, 1]) and
                        ([next_ob,1], random_action, [next_ob, 1])
                        This way we can handle varying types of terminal states.
                        """
                        # next_ob is the absorbing state
                        # for now just taking the previous action
                        self._handle_step(
                            next_ob,
                            # action,
                            self.training_env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            np.zeros_like(next_ob),
                            np.array([False]),
                            timeout,
                            pred_obs=pred_obs_prime,
                            absorbing=np.array([0.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info,
                            action_log_prob=act_log_prob,
                        )
                        self._handle_step(
                            np.zeros_like(next_ob),
                            # action,
                            self.training_env.action_space.sample(),
                            # the reward doesn't matter
                            reward,
                            # next_ob,
                            np.zeros_like(next_ob),
                            np.array([False]),
                            timeout,
                            pred_obs=pred_obs_prime,
                            absorbing=np.array([1.0, 1.0]),
                            agent_info=agent_info,
                            env_info=env_info,
                            action_log_prob=act_log_prob,
                        )
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                elif len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

            if (self.save_buffer_data) and (self.num_epochs % 10 == 0):
                assert self.save_path != None
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.replay_buffer.save_data(
                    os.path.join(
                        self.save_path, "buffer_data_epoch_{}.pkl".format(epoch)
                    )
                )
