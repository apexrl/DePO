from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        inverse_mode="MLE",
        state_predictor_mode="MSE",
        update_both=True,
        train_alpha=True,
        reward_scale=1.0,
        discount=0.99,
        alpha=0.2,
        policy_lr=1e-3,
        qf_lr=1e-3,
        alpha_lr=3e-4,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        target_sp=10,
        sp_alpha=1.0,
        train_sp_alpha=True,
        **kwargs
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.state_predictor_mode = state_predictor_mode
        self.inverse_mode = inverse_mode

        self.train_alpha = train_alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=train_alpha)
        self.alpha = self.log_alpha.detach().exp()
        assert "env" in kwargs.keys(), "env info should be taken into SAC alpha"
        self.target_entropy = -np.prod(kwargs["env"].action_space.shape)
        print("TARGET SP ALPHA: {}".format(target_sp))
        self.target_sp = target_sp
        self.train_sp_alpha = train_sp_alpha

        self.log_sp_alpha = torch.tensor(np.log(sp_alpha), requires_grad=train_alpha)
        self.sp_alpha = self.log_sp_alpha.detach().exp()

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.update_both = update_both
        if not update_both:
            print("\n\nSAC UPDATE SP ONLY!\n\n")
            self.policy_optimizer = optimizer_class(
                self.policy.state_predictor.parameters(),
                lr=policy_lr,
                betas=(beta_1, 0.999),
            )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha], lr=alpha_lr, betas=(beta_1, 0.999)
        )
        self.sp_alpha_optimizer = optimizer_class(
            [self.log_sp_alpha], lr=alpha_lr, betas=(beta_1, 0.999)
        )

    def train_step(
        self,
        batch,
        qss=False,
        alpha=None,
        beta=None,
        expert_batch=None,
        inv_batch=None,
        state_diff=False,
        multi_step=False,
        step_num=1,
        target_state_predictor=None,
        cycle=False,
        forward_model=None,
        cycle_weight=1.0,
    ):
        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        input_act = actions
        if qss:
            input_act = next_obs
        """
        QF Loss
        """
        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, input_act)
        q2_pred = self.qf2(obs, input_act)

        # Make sure policy accounts for squashing functions like tanh correctly!
        next_policy_outputs = self.policy(
            next_obs, return_log_prob=True, return_predicting_obs=qss
        )
        if qss:
            (
                next_new_next_obs,
                next_new_actions,
                next_policy_mean,
                next_policy_log_std,
                next_log_pi,
            ) = next_policy_outputs[:5]
        else:
            (
                next_new_actions,
                next_policy_mean,
                next_policy_log_std,
                next_log_pi,
            ) = next_policy_outputs[:4]
        next_new_input_act = next_new_actions
        if qss:
            next_new_input_act = next_new_next_obs
        target_qf1_values = self.target_qf1(
            next_obs, next_new_input_act
        )  # do not need grad || it's the shared part of two calculation
        target_qf2_values = self.target_qf2(
            next_obs, next_new_input_act
        )  # do not need grad || it's the shared part of two calculation
        min_target_value = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * (
            min_target_value - self.alpha * next_log_pi
        )  ## original implementation has detach
        q_target = q_target.detach()

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy Loss
        """
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(self.alpha * log_pi - q_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        if (beta is not None) and (self.update_both):
            print("do not update beta inv loss")
            exit(0)
            inv_obs = inv_batch["observations"]
            inv_next_obs = inv_batch["next_observations"]
            inv_actions = inv_batch["actions"]

            if self.inverse_mode == "MLE":
                log_prob = self.policy.inverse_dynamic.get_log_prob(
                    inv_obs, inv_next_obs, inv_actions
                )
                id_loss = -1.0 * log_prob.mean()
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics[
                    "Inverse-Dynamic-Log-Likelihood"
                ] = ptu.get_numpy(-1.0 * id_loss)

                assert not torch.max(
                    torch.isnan(id_loss)
                ), "nan-inverse-dynamic-training, obs: {}, obs_prime: {}, acts: {}, log_std: {}".format(
                    inv_obs, inv_next_obs, inv_actions, log_prob
                )
            elif self.inverse_mode == "MSE":
                pred_acts = self.policy.inverse_dynamic(inv_obs, inv_next_obs)[0]
                squared_diff = (pred_acts - inv_actions) ** 2
                id_loss = torch.sum(squared_diff, dim=-1).mean()

                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["Inverse-Dynamic-MSE"] = ptu.get_numpy(id_loss)

            policy_loss = policy_loss + beta * id_loss

        if alpha is not None:
            exp_obs = expert_batch["observations"]
            exp_next_obs = expert_batch["next_observations"]

            if self.state_predictor_mode == "MLE":
                label_obs = exp_next_obs
                if state_diff:
                    label_obs = exp_next_obs - exp_obs
                log_prob = self.policy.state_predictor.get_log_prob(exp_obs, label_obs)
                sp_loss = -1.0 * log_prob.mean()
                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Predictor-Log-Likelihood"] = ptu.get_numpy(
                    -1.0 * sp_loss
                )

                if cycle:
                    pred_act = self.policy(obs)[0].detach()
                    model_input = torch.cat([obs, pred_act], dim=-1)
                    forward_model_pred_next_state = (
                        forward_model(model_input).detach() + obs
                    )
                    cycle_log_prob = self.policy.state_predictor.get_log_prob(
                        obs, forward_model_pred_next_state
                    )
                    sp_loss = sp_loss + cycle_weight * cycle_log_prob.mean()

                # pred_obs = self.policy.state_predictor(exp_obs)
                pred_obs = self.policy.state_predictor(exp_obs, deterministic=True)[0]
                squared_diff = (pred_obs - label_obs) ** 2

                expt_mse = torch.sum(squared_diff, dim=-1).mean()
                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Pred-Expt-MSE"] = ptu.get_numpy(expt_mse)

            if self.state_predictor_mode == "MSE":
                # raise ValueError("not MSE!")
                # pred_obs = self.policy.state_predictor(exp_obs)
                pred_obs = self.policy.state_predictor(exp_obs, deterministic=True)[0]
                label_obs = exp_next_obs
                if state_diff:
                    label_obs = exp_next_obs - exp_obs
                squared_diff = (pred_obs - label_obs) ** 2
                sp_loss = torch.sum(squared_diff, dim=-1)

                if multi_step:
                    next_pred_obs = pred_obs
                    for i in np.arange(1, step_num + 1):
                        pred_obs_use = next_pred_obs
                        next_i_obs = expert_batch["next{}_observations".format(i)]
                        next_pred_obs = target_state_predictor(pred_obs_use)
                        squared_diff = (next_pred_obs - next_i_obs) ** 2
                        sp_loss = sp_loss + torch.sum(squared_diff, dim=-1)
                if cycle:
                    pred_act = self.policy(obs)[0].detach()
                    model_input = torch.cat([obs, pred_act], dim=-1)
                    forward_model_pred_next_state = (
                        forward_model(model_input).detach() + obs
                    )
                    squared_diff_3 = (pred_obs - forward_model_pred_next_state) ** 2
                    sp_loss = sp_loss + cycle_weight * torch.sum(squared_diff_3, dim=-1)

                sp_loss = sp_loss.mean()

                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Pred-Expt-MSE"] = ptu.get_numpy(sp_loss)

                agent_label_obs = next_obs
                if state_diff:
                    agent_label_obs = next_obs - obs
                agent_pred_obs = self.policy.state_predictor(obs)[0]
                agent_squared_diff = (agent_pred_obs - agent_label_obs) ** 2
                agent_loss = torch.sum(agent_squared_diff, dim=-1).mean()
                self.eval_statistics["State-Pred-Real-MSE"] = ptu.get_numpy(agent_loss)

            policy_loss = policy_loss + self.sp_alpha * sp_loss

        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update alpha
        """
        if self.train_alpha:
            log_prob = log_pi.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        """
        Update sp alpha
        """
        if self.train_sp_alpha:
            if self.state_predictor_mode == "MSE":
                sp = self.target_sp - sp_loss.detach()
            elif self.state_predictor_mode == "MLE":
                sp = sp_loss.detach() - self.target_sp
            sp_alpha_loss = (self.log_sp_alpha * sp).mean()
            self.sp_alpha_optimizer.zero_grad()
            sp_alpha_loss.backward()
            self.sp_alpha_optimizer.step()
            self.sp_alpha = self.log_sp_alpha.detach().exp()

        """
        Update networks
        """

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

        self.eval_statistics["Reward Scale"] = self.reward_scale
        self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
        self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
        self.eval_statistics["SP-Alpha Loss"] = np.mean(ptu.get_numpy(sp_alpha_loss))
        self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q1 Predictions",
                ptu.get_numpy(q1_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q2 Predictions",
                ptu.get_numpy(q2_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Alpha",
                [ptu.get_numpy(self.alpha)],
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "SP-Alpha",
                [ptu.get_numpy(self.sp_alpha)],
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Log Pis",
                ptu.get_numpy(log_pi),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Policy mu",
                ptu.get_numpy(policy_mean),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Policy log std",
                ptu.get_numpy(policy_log_std),
            )
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            log_alpha=self.log_alpha,
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
