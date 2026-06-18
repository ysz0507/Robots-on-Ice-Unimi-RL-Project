import abc
from dataclasses import dataclass

import numpy as np
from torch import nn


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select_action(self, state):
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_models(self) -> dict[str, nn.Module]:
        pass
