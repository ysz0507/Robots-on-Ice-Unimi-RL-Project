import random

import numpy as np
import pygame

from settings import RenderingSettings


class StationaryEntity:
    def __init__(self, x, y, image_path, width):
        self.pos = np.array([x, y], dtype=np.float32)
        self.background = self._load_image(image_path, width)

    @staticmethod
    def _load_image(path, width):
        if not pygame.get_init():
            return None
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale_by(
            image,
            width / image.get_width()
        )

    def draw(self, screen):
        screen.blit(self.background, self.pos - self.background.get_rect().center)

    def reset(self, x, y):
        self.pos[:] = [x, y]


class Robot(StationaryEntity):
    def __init__(self, x, y, mass):
        super().__init__(x, y, "assets/robot.png", RenderingSettings.ROBOT_WIDTH)
        self.mass = mass
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, x, y):
        super().reset(x, y)
        self.vel[:] = [0.0, 0.0]

    def update(self, force):
        """
        Physics update:

        x(t+1) = x(t) + dt * v(t)
        v(t+1) = v(t) + dt * F(t)/m
        """

        # Clamp force magnitude
        norm = np.linalg.norm(force)
        if norm > RenderingSettings.MAX_FORCE:
            force = force / norm * RenderingSettings.MAX_FORCE

        # Position update
        self.pos += RenderingSettings.DT * self.vel

        # Velocity update
        self.vel += RenderingSettings.DT * (force / self.mass)
        self.vel -= RenderingSettings.ICE_FRICTION * self.vel


class Target(StationaryEntity):
    def __init__(self):
        pos = self.__random_position()
        super().__init__(pos[0], pos[1], self.__random_star_path(), RenderingSettings.TARGET_WIDTH)

    @staticmethod
    def __random_star_path():
        index = random.randint(1, 4)
        return f"assets/star_{index}.png"

    @staticmethod
    def __random_position():
        margin = 50

        x = random.randint(margin, RenderingSettings.WIDTH - margin)
        y = random.randint(margin, RenderingSettings.HEIGHT - margin)

        return np.array([x, y], dtype=np.float32)

    def respawn(self):
        self.background = self._load_image(self.__random_star_path(), RenderingSettings.TARGET_WIDTH)
        self.pos = self.__random_position()
