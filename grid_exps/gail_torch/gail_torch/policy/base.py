from torch import nn
import torch
from abc import ABC, abstractmethod


class BasePolicy(ABC, nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        discriminator=None,
        device=torch.device("cpu"),
        writer=None,
    ):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.writer = writer
        self.device = device
        self.discriminator = discriminator
        self._cnt = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["writer"]
        return state

    def set_device(self, device):
        self.device = device
        self = self.to(device)

    @abstractmethod
    def update(self, memory):
        pass

    @abstractmethod
    def get_action(self, obs):
        pass
