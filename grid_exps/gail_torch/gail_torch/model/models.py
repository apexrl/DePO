import gym
import torch
import torch.nn as nn

from gail_torch.utils import normal_log_density
from gail_torch.model.spectral_norm_fc import spectral_norm_fc


class discrete_actor(nn.Module):
    def __init__(self, state_dim, num_actions, num_units):
        super(discrete_actor, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(state_dim, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, num_actions)

        self.train()

    def get_log_prob(self, x, actions):
        action_prob = torch.softmax(self.forward(x), dim=1)
        return torch.log(
            action_prob.gather(1, torch.where(actions > 0)[1].unsqueeze(1))
        )

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


class discrete_critic(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(discrete_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


class discrete_qnet(nn.Module):
    def __init__(self, num_inputs, num_actions, num_units):
        super(discrete_qnet, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, num_actions)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


# TODO(zbzhu): change discriminator_net to unified mlp class
class discriminator_net(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(discriminator_net, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))
        prob = torch.sigmoid(self.fc_out(x))
        return prob


class continuous_actor(nn.Module):
    def __init__(self, num_inputs, action_dim, num_units, log_std=0, activation=None):
        super(continuous_actor, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, action_dim)
        if activation == "tanh":
            self.activation_func = nn.Tanh()
        elif activation is None:
            self.activation_func = None

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        self.train()

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        if self.activation_func is None:
            action_mean = self.fc_out(x)
        else:
            action_mean = self.activation_func(self.fc_out(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std


class continuous_critic(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(continuous_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


class iresnet_block(nn.Module):
    def __init__(
        self,
        dims,
        input_act=False,
        coeff=0.97,
        n_power_iter=5,
        activation=nn.LeakyReLU(0.01),
    ):
        super(iresnet_block, self).__init__()
        self.activation = activation
        self.coeff = coeff
        self.n_power_iter = n_power_iter

        layers = []
        if input_act:
            layers.append(activation)
        layers.append(self._wrapper_spectral_norm(nn.Linear(dims, dims)))
        layers.append(activation)
        layers.append(self._wrapper_spectral_norm(nn.Linear(dims, dims)))
        layers.append(activation)
        layers.append(self._wrapper_spectral_norm(nn.Linear(dims, dims)))

        self.block = nn.Sequential(*layers)

        self.train()

    def forward(self, x):
        # print(x.shape)
        # print(self.block(x).shape)
        x = self.block(x) + x
        return x

    def _wrapper_spectral_norm(self, layer):
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        return spectral_norm_fc(layer, self.coeff, n_power_iterations=self.n_power_iter)


class iresnet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_blocks,
        num_output,
        coeff=0.97,
        n_power_iter=5,
        activation=nn.LeakyReLU(0.01),
    ):
        super(iresnet, self).__init__()
        self.activation = activation
        self.coeff = coeff
        self.n_power_iter = n_power_iter

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                iresnet_block(num_inputs, i > 0, coeff, n_power_iter, activation)
            )

        self.logits = nn.Linear(num_inputs, num_output)

        self.train()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        action_logits = self.logits(self.activation(x))
        return action_logits

    # def to(self, device):
    #     super().to(device)
    #     for block in self.blocks:
    #         block.to(device)
