import abc
from dataclasses import dataclass
from typing import Sequence

import numpy as np


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
    def train(self, transitions: list[Transition]) -> tuple[float, float]:
        pass

    @abc.abstractmethod
    def get_models(self) -> Sequence:
        pass
