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
        vf,
        inverse_mode="MLE",
        state_predictor_mode="MSE",
        update_both=True,
        train_alpha=True,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        **kwargs
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.state_predictor_mode = state_predictor_mode
        self.inverse_mode = inverse_mode

        self.target_vf = vf.copy()

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
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), lr=vf_lr, betas=(beta_1, 0.999)
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
        target_v_values = self.target_vf(
            next_obs
        )  # do not need grad || it's the shared part of two calculation
        q_target = (
            rewards + (1.0 - terminals) * self.discount * target_v_values
        )  ## original implementation has detach
        q_target = q_target.detach()
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        """
        VF Loss
        """
        self.vf_optimizer.zero_grad()
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(
            obs, return_log_prob=True, return_predicting_obs=qss
        )
        # in this part, we only need new_actions and log_pi with no grad
        if qss:
            (
                new_next_obs,
                new_actions,
                policy_mean,
                policy_log_std,
                log_pi,
            ) = policy_outputs[:5]
        else:
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        new_input_act = new_actions
        if qss:
            new_input_act = new_next_obs
        q1_new_acts = self.qf1(obs, new_input_act)
        q2_new_acts = self.qf2(obs, new_input_act)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)
        v_target = q_new_actions - log_pi
        v_target = v_target.detach()
        vf_loss = 0.5 * torch.mean((v_pred - v_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()
        vf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self.vf_optimizer.step()

        """
        Policy Loss
        """
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(log_pi - q_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        if (beta is not None) and (self.update_both):
            print("update beta inv loss")
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

            if self.state_predictor_mode == "MSE":
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

            elif self.state_predictor_mode == "MLE":
                log_prob = self.policy.state_predictor.get_log_prob(
                    exp_obs, exp_next_obs
                )
                sp_loss = -1.0 * log_prob.mean()
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics[
                    "State-Predictor-Log-Likelihood"
                ] = ptu.get_numpy(-1.0 * sp_loss)

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

            policy_loss = policy_loss + alpha * sp_loss

        policy_loss.backward()
        self.policy_optimizer.step()

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
            self.eval_statistics["VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
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
            self.vf,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
