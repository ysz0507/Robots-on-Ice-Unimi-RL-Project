from typing import Sequence

import numpy as np
import pygame

from agents.agent import Agent, Transition
from settings import RenderingSettings


class HumanAgent(Agent):
    def get_models(self) -> Sequence:
        return ()

    def train(self, transitions: list[Transition]) -> tuple[float, float]:
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