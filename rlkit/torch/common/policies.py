import numpy as np
from numpy.random import choice

import torch
from torch import nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.common.networks import Mlp, FlattenMlp
from rlkit.torch.core import PyTorchModule

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.serializable import Serializable

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, **kwargs):
        return self.stochastic_policy.get_action(
            **kwargs,
            deterministic=True,
        )

    def get_actions(self, **kwargs):
        return self.stochastic_policy.get_actions(
            **kwargs,
            deterministic=True,
        )

    def train(self, mode):
        pass

    def set_num_steps_total(self, num):
        pass

    def to(self, device):
        self.stochastic_policy.to(device)


class DiscretePolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = DiscretePolicy(...)
    action, log_prob = policy(obs, return_log_prob=True)
    ```
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=nn.LogSoftmax(1),
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        action = self.get_actions(obs_np[None], deterministic=deterministic)
        return action, {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        log_probs, pre_act = super().forward(obs, return_preactivations=True)

        if deterministic:
            log_prob, idx = torch.max(log_probs, 1)
            return (idx, None)
        else:
            # Using Gumbel-Max trick to sample from the multinomials
            u = torch.rand(pre_act.size(), requires_grad=False)
            gumbel = -torch.log(-torch.log(u))
            _, idx = torch.max(gumbel + pre_act, 1)

            idx = torch.unsqueeze(idx, 1)
            log_prob = torch.gather(log_probs, 1, idx)

            # # print(log_probs.size(-1))
            # # print(log_probs.data.numpy())
            # # print(np.exp(log_probs.data.numpy()))
            # idx = choice(
            #     log_probs.size(-1),
            #     size=1,
            #     p=np.exp(log_probs.data.numpy())
            # )
            # log_prob = log_probs[0,idx]

            # print(idx)
            # print(log_prob)

            return (idx, log_prob)

    def get_log_pis(self, obs):
        return super().forward(obs)


class MlpPolicy(Mlp, ExplorationPolicy):
    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )

    def get_action(self, obs_np, deterministic=False):
        """
        deterministic=False makes no diff, just doing this for
        consistency in interface for now
        """
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        actions = actions[None]
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np)[0]


class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3, **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # print('forward')
        # print(obs.shape)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        # print('fuck')
        # print(h)
        mean = self.last_fc(h)
        # print(mean)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
            # print('mean, std')
            # print(mean)
            # print(log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        # I'm doing it like this for now for backwards compatibility, sorry!
        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        # print('\n\n\n\n\nGet log prob')
        # print(log_prob)
        # print(mean)
        # print(log_std)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3, **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            assert LOG_SIG_MIN <= np.log(std) <= LOG_SIG_MAX
            std = std * np.ones((1, action_dim))
            self.log_std = ptu.from_numpy(np.log(std), requires_grad=False)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # print('forward')
        # print(obs)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        # print('fuck')
        # print(h)
        mean = self.last_fc(h)

        # print(mean)
        if self.std is None:
            log_std = self.last_fc_log_std(h)

            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, np.log(0.2))

            std = torch.exp(log_std)

        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            # print(mean, log_std)
            action = normal.sample()
            if return_log_prob:
                log_prob = normal.log_prob(action)

        # I'm doing it like this for now for backwards compatibility, sorry!
        if return_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamTanhMultivariateGaussianLfOPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        sp_hidden_sizes,
        inv_hidden_sizes,
        state_predictor=None,
        inverse_dynamic=None,
        sample_num=1,
        init_w=1e-3,
        state_diff=False,
        deterministic_sp=False,
        deterministic_inv=False,
        spectral_norm_inv=False,
        inv_noise=False,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()

        self.state_predictor = state_predictor
        self.inverse_dynamic = inverse_dynamic

        if state_predictor is None:
            # self.state_predictor = FlattenMlp(
            #     hidden_sizes=hidden_sizes,
            #     input_size=obs_dim,
            #     output_size=obs_dim,
            #     init_w=init_w,
            # )
            self.state_predictor = ReparamMultivariateGaussianPolicy(
                hidden_sizes=sp_hidden_sizes,
                obs_dim=obs_dim,
                action_dim=obs_dim,
                init_w=init_w,
            )
        if inverse_dynamic is None:
            self.inverse_dynamic = InverseDynamic(
                hidden_sizes=inv_hidden_sizes,
                input_size=obs_dim * 2,
                output_size=action_dim,
                init_w=init_w,
                spectral_norm=spectral_norm_inv,
                noise=inv_noise,
            )

        assert self.state_predictor is not None
        assert self.inverse_dynamic is not None

        self.deterministic_sp = deterministic_sp
        self.deterministic_inv = deterministic_inv

        self.state_diff = state_diff
        self.sample_num = sample_num

    def get_action_by_next_state(self, obs_np, obs_prime_np, deterministic=False):
        actions = self.inverse_dynamic.get_action(
            obs_np, obs_prime_np, deterministic=deterministic
        )
        return actions[0, :], {}

    def get_action(self, obs_np, deterministic=False, return_predicting_obs=False):
        actions = self.get_actions(
            obs_np[None],
            deterministic=deterministic,
            return_predicting_obs=return_predicting_obs,
        )
        # print(actions, actions.shape)
        if not return_predicting_obs:
            return actions[0, :], {}
        else:
            pred_obs, actions = actions
            return pred_obs[0, :], actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False, return_predicting_obs=False):
        if return_predicting_obs:
            return self.eval_np(
                obs_np,
                deterministic=deterministic,
                return_predicting_obs=return_predicting_obs,
            )[:2]
        return self.eval_np(
            obs_np,
            deterministic=deterministic,
            return_predicting_obs=return_predicting_obs,
        )[0]

    def get_next_states(self, obs_np, deterministic=False):
        return self.state_predictor.eval_np(obs_np)[0]

    def set_num_steps_total(self, t):
        self.inverse_dynamic.set_num_steps_total(t)

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        return_predicting_obs=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        if not deterministic and self.sample_num > 1 and return_log_prob:
            assert len(obs.shape) == 2, "only for vector feature!"
            batch_size = obs.shape[0]
            obs = obs.repeat([self.sample_num, 1])

        (obs_prime, sp_mean, sp_log_std, sp_log_prob) = self.state_predictor(
            obs,
            deterministic=(self.deterministic_sp or deterministic),
            return_log_prob=return_log_prob,
        )[:4]

        if self.state_diff:
            obs_prime += obs

        (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        ) = self.inverse_dynamic(
            obs,
            obs_prime,
            (self.deterministic_inv or deterministic),
            return_log_prob,
            return_tanh_normal=False,
        )

        # assert not torch.max(
        #     torch.isnan(action)
        # ), "nan0, obs: {}, obs_prime: {}".format(obs, obs_prime)

        # if self.deterministic_inv:
        #     log_std = sp_log_std
        #     log_prob = sp_log_prob

        if not deterministic and self.sample_num > 1 and return_log_prob:
            obs_prime = obs_prime.reshape([-1, batch_size, obs_prime.shape[-1]])[0]
            action = action.reshape([-1, batch_size, action.shape[-1]])[0]
            log_std = log_std.reshape([-1, batch_size, 1]).mean(axis=0)
            log_prob = log_prob.reshape([-1, batch_size, 1]).exp().mean(axis=0).log()

        if sp_log_prob is None:
            sp_log_prob = torch.Tensor([0.0]).to(ptu.device)

        if return_predicting_obs:
            return (
                obs_prime,
                action,
                mean,
                # sp_log_std,
                # sp_log_prob,
                log_std,
                log_prob,
                # sp_log_prob+log_prob if return_log_prob else None,
                # sp_log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
            )

        return (
            action,
            mean,
            # sp_log_std,
            # sp_log_prob,
            log_std,
            log_prob,
            # sp_log_prob+log_prob if return_log_prob else None,
            # sp_log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    def get_log_prob(self, obs, acts, return_normal_params=False):
        if self.sample_num > 1:
            assert len(obs.shape) == 2, "only for vector feature!"
            batch_size = obs.shape[0]
            obs = obs.repeat([self.sample_num, 1])

        (obs_prime, sp_mean, sp_log_std, sp_log_prob) = self.state_predictor(
            obs, deterministic=False, return_log_prob=True
        )[:4]

        if self.state_diff:
            obs_prime += obs

        (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        ) = self.inverse_dynamic(
            obs,
            obs_prime,
            deterministic=False,
            return_log_prob=True,
            return_tanh_normal=False,
        )

        if self.sample_num > 1:
            obs_prime = obs_prime.reshape([-1, batch_size, obs_prime.shape[-1]])[0]
            action = action.reshape([-1, batch_size, action.shape[-1]])[0]
            log_std = log_std.reshape([-1, batch_size, 1]).mean(axis=0)
            log_prob = log_prob.reshape([-1, batch_size, 1]).exp().mean(axis=0).log()

        if sp_log_prob is None:
            sp_log_prob = torch.Tensor([0.0]).to(ptu.device)

        return sp_log_prob + log_prob


class InverseDynamic(Mlp):
    def __init__(
        self,
        hidden_sizes,
        input_size,
        output_size,
        init_w=1e-3,
        std=None,
        noise=False,
        max_sigma=0.1,
        min_sigma=0.1,
        decay_period=1000000,
        max_act=1.0,
        min_act=-1.0,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            init_w=init_w,
            **kwargs,
        )

        if self.spectral_norm:
            print("\n SPECTUAL NORM INV DYNAMICS!")

        self.log_std = None
        self.std = std
        self.noise = noise

        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._decay_period = decay_period
        self.max_act = max_act
        self.min_act = min_act
        self.t = 0
        if std is None:
            last_hidden_size = input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, output_size)
            if self.spectral_norm:
                self.last_fc_log_std = SpectralNorm(self.last_fc_log_std)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, obs_prime_np, deterministic=False):
        actions = self.get_actions(
            obs_np[None], obs_prime_np[None], deterministic=deterministic
        )
        return actions[0, :], {}

    def get_actions(self, obs_np, obs_prime_np, deterministic=False):
        return self.eval_np(obs_np, obs_prime_np, deterministic=deterministic)[0]

    def set_num_steps_total(self, t):
        self.t = t

    def forward(
        self,
        obs,
        obs_prime,
        deterministic=False,
        return_log_prob=False,
        return_tanh_normal=False,
    ):
        h = torch.cat([obs, obs_prime], axis=-1)
        for i, fc in enumerate(self.fcs):
            assert not torch.max(
                torch.isnan(h)
            ), "nan-inverse-dynamic-net, i: {}".format(i)
            h = self.hidden_activation(fc(h))

        mean = self.last_fc(h)
        # print(mean)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
            # print('mean, std')
            # print(mean)
            # print(log_std)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        if self.noise:
            sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
                1.0, self.t * 1.0 / self._decay_period
            )
            action = torch.clamp(
                action + torch.normal(torch.zeros_like(action), sigma),
                self.min_act,
                self.max_act,
            )

        # assert not torch.max(
        #     torch.isnan(action)
        # ), "nan1, obs: {}, obs_prime: {}, mu: {}, log_std: {}, h:{}".format(
        #     obs, obs_prime, mean, log_std, h
        # )

        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    def get_log_prob(self, obs, obs_prime, acts, return_normal_params=False):
        h = torch.cat([obs, obs_prime], axis=-1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        assert not torch.max(
            torch.isnan(log_prob)
        ), "nan2, s:{}, s':{}, acts:{}, mu: {}, log_std: {}".format(
            obs, obs_prime, acts, mean, log_std
        )

        # print('\n\n\n\n\nGet log prob')
        # print(log_prob)
        # print(mean)
        # print(log_std)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamTanhMultivariateGaussianEncoderPolicy(
    ReparamTanhMultivariateGaussianPolicy
):
    """
    Policy with encoder
    Usage:
    ```
    policy = ReparamTanhMultivariateGaussianEncoderPolicy(...)
    """

    def __init__(self, encoder, **kwargs):
        self.save_init_params(locals())
        super().__init__(**kwargs)
        self.encoder = encoder

    def forward(self, obs, use_feature=False, **kwargs):
        """
        :param obs: Observation
        """
        feature_obs = obs
        if not use_feature:
            feature_obs = self.encoder(obs)
        return super().forward(feature_obs, **kwargs)
