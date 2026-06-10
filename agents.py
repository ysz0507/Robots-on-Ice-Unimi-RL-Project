import abc
from dataclasses import dataclass

import numpy as np
import pygame

from settings import RenderingSettings


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
    def train(self, transitions: list[Transition]):
        pass


class RLAgent(Agent):
    def select_action(self, state):
        """
        Replace with policy network output.
        """

        # Random continuous action
        force = np.random.uniform(
            low=-RenderingSettings.MAX_FORCE,
            high=RenderingSettings.MAX_FORCE,
            size=(2,)
        )

        return force

    def train(self, transitions: list[Transition]):
        """
        TODO: Replace with training logic.
        Note: the transitions are not scaled.
        """
        pass


class HumanAgent(Agent):
    def train(self, transitions: list[Transition]):
        pass

    def __init__(self):
        self.key_mapping = {
            pygame.K_UP: (0.0, -RenderingSettings.MAX_FORCE),
            pygame.K_DOWN: (0.0, RenderingSettings.MAX_FORCE),
            pygame.K_LEFT: (-RenderingSettings.MAX_FORCE, 0.0),
            pygame.K_RIGHT: (RenderingSettings.MAX_FORCE, 0.0)
        }
        if not pygame.get_init():
            raise RuntimeError("Pygame must be initialized for HumanAgents.")

    def select_action(self, state):
        keys = pygame.key.get_pressed()

        force = np.array([0.0, 0.0], dtype=np.float32)

        for key, action in self.key_mapping.items():
            if keys[key]:
                force += action

        # Clamp force magnitude
        norm = np.linalg.norm(force)
        if norm > RenderingSettings.MAX_FORCE:
            force = force / norm * RenderingSettings.MAX_FORCE

        return force