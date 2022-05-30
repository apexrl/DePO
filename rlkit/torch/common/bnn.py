import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.common.distributions import (
    ReparamMultivariateNormalDiag,
    ReparamTanhMultivariateNormal,
)
import itertools
import torch.optim as optim
from rlkit.torch.common.networks import identity


class StandardScaler(object):
    def __init__(self):
        self.mu = 0.0
        self.std = 1.0
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        output_activation=identity,
        optimizer_class=optim.Adam,
    ):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(
            input_size, hidden_size, ensemble_size, weight_decay=0.000025
        )
        self.nn2 = EnsembleFC(
            hidden_size, hidden_size, ensemble_size, weight_decay=0.00005
        )
        self.nn3 = EnsembleFC(
            hidden_size, hidden_size, ensemble_size, weight_decay=0.000075
        )
        self.nn4 = EnsembleFC(
            hidden_size, hidden_size, ensemble_size, weight_decay=0.000075
        )
        self.use_decay = use_decay
        self.output_activation = output_activation

        self.output_dim = output_size
        # Add variance output
        self.nn5 = EnsembleFC(
            hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001
        )

        self.max_logvar = nn.Parameter(
            (torch.ones((1, self.output_dim)).float() / 2).to(ptu.device),
            requires_grad=False,
        )
        self.min_logvar = nn.Parameter(
            (-torch.ones((1, self.output_dim)).float() * 10).to(ptu.device),
            requires_grad=False,
        )
        self.optimizer = optimizer_class(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.output_activation(self.nn5(nn4_output))

        mean = nn5_output[:, :, : self.output_dim]

        logvar = self.max_logvar - F.softplus(
            self.max_logvar - nn5_output[:, :, self.output_dim :]
        )
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.0
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.0
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # print(inc_var_loss,mean.shape)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1
            )
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def do_train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class EnsembleInverseDynamicsModel(PyTorchModule):
    def __init__(
        self,
        num_networks,
        num_elites,
        state_size,
        action_size,
        hidden_size=256,
        use_decay=False,
        holdout_ratio=0.0,
        max_epochs_since_update=5,
        max_logging=5000,
        optimizer_class=optim.Adam,
        learning_rate=1e-3,
        max_sigma=0.1,
        min_sigma=0.1,
        decay_period=1000000,
        max_act=1.0,
        min_act=-1.0,
        noise=True,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.network_size = num_networks
        self.elite_size = num_elites
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.elite_model_idxes = []
        self.holdout_ratio = holdout_ratio
        self.ensemble_model = EnsembleModel(
            state_size * 2,
            action_size,
            num_networks,
            hidden_size,
            use_decay=use_decay,
            output_activation=torch.tanh,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
        )
        self.scaler = StandardScaler()
        self._max_epochs_since_update = max_epochs_since_update
        self.max_logging = max_logging

        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._decay_period = decay_period
        self.max_act = max_act
        self.min_act = min_act
        self.t = 0
        self.noise = noise

    def set_num_steps_total(self, t):
        self.t = t

    def do_train(self, inputs, labels, batch_size=256):
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = min(int(inputs.shape[0] * self.holdout_ratio), self.max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(ptu.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(ptu.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack(
                [
                    np.random.permutation(train_inputs.shape[0])
                    for _ in range(self.network_size)
                ]
            )
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos : start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(ptu.device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(ptu.device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.do_train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(
                    holdout_inputs, ret_log_var=True
                )
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[: self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            # print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))

        return np.mean(holdout_mse_losses)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def forward(
        self, obs, obs_prime, deterministic=False, return_log_prob=False, **kwargs
    ):
        obs = obs.detach().cpu().numpy()
        obs_prime = obs_prime.detach().cpu().numpy()
        input_data = np.concatenate([obs, obs_prime], axis=-1)
        mean = action = torch.Tensor(self.predict(input_data, factored=False)[0]).to(
            ptu.device
        )

        if not deterministic or self.noise:
            self.sigma = sigma = self._max_sigma - (
                self._max_sigma - self._min_sigma
            ) * min(1.0, self.t * 1.0 / self._decay_period)
            action = torch.clamp(
                mean + torch.normal(torch.zeros_like(action), sigma),
                self.min_act,
                self.max_act,
            )
        if deterministic:
            action = mean

        log_prob = None
        log_std = None
        std = None
        expected_log_prob = None
        mean_action_log_prob = None
        if return_log_prob:
            sigma = (torch.ones_like(mean) * self.sigma).to(ptu.device)
            tanh_normal = ReparamTanhMultivariateNormal(mean, sigma)
            log_prob = tanh_normal.log_prob(action)

            expected_log_prob = mean_action_log_prob = log_prob = torch.ones(
                [input_data.shape[0], 1]
            ).to(ptu.device)
            std = torch.zeros([input_data.shape[0], 1]).to(ptu.device)

        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            None,
        )

    def get_log_prob(self, obs, obs_prime, action, return_normal_params=False):
        obs = obs.cpu().numpy()
        obs_prime = obs.cpu().numpy()

        input_data = np.concatenate([obs, obs_prime], axis=-1)
        mean = action = torch.Tensor(self.predict(input_data, factored=False)[0]).to(
            ptu.device
        )

        sigma = (torch.ones_like(mean) * self.sigma).to(ptu.device)
        tanh_normal = ReparamTanhMultivariateNormal(mean, sigma)
        log_prob = tanh_normal.log_prob(action)

        if return_normal_params:
            return log_prob, mean, sigma

        return log_prob

    def predict(self, inputs, batch_size=1024, factored=False):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = (
                torch.from_numpy(inputs[i : min(i + batch_size, inputs.shape[0])])
                .float()
                .to(ptu.device)
            )
            b_mean, b_var = self.ensemble_model(
                input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False
            )
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)
        # print(input.shape,ensemble_mean.shape)
        if factored:
            return ensemble_mean, ensemble_var
        else:
            #             assert False, "Need to transform to numpy"
            #             print(ensemble_mean, ensemble_mean.shape)
            mean = np.mean(ensemble_mean, axis=0)
            var = np.mean(ensemble_var, axis=0) + np.mean(
                np.square(ensemble_mean - mean[None, :, :]), axis=0
            )
            return mean, var


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
