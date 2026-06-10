import numpy as np
import pygame

from settings import Settings

class TrainedAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        """
        Replace with policy network output.
        """

        # Random continuous action
        force = np.random.uniform(
            low=-Settings.MAX_FORCE,
            high=Settings.MAX_FORCE,
            size=(2,)
        )

        return force

    def train(self):
        """
        Placeholder for policy gradient update.
        """
        pass

class HumanAgent:
    def __init__(self):
        self.key_mapping = {
            pygame.K_UP: (0.0, -Settings.MAX_FORCE),
            pygame.K_DOWN: (0.0, Settings.MAX_FORCE),
            pygame.K_LEFT: (-Settings.MAX_FORCE, 0.0),
            pygame.K_RIGHT: (Settings.MAX_FORCE, 0.0)
        }

    def select_action(self, state):
        keys = pygame.key.get_pressed()

        force = np.array([0.0, 0.0], dtype=np.float32)

        for key, action in self.key_mapping.items():
            if keys[key]:
                force += action

        return force