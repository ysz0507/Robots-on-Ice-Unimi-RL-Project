import random

import numpy as np
import pygame

from settings import RenderingSettings


class StationaryEntity:
    def __init__(self, x, y, image_path=None, width=None):
        self.pos = np.array([x, y], dtype=np.float32)
        if image_path is not None:
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
        screen.blit(self.background, 100 * self.pos - self.background.get_rect().center)

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
        norm = np.linalg.norm(force)  # Newtons
        if norm > RenderingSettings.MAX_FORCE:
            force = force / norm * RenderingSettings.MAX_FORCE

        a = force / self.mass  # Acceleration in m/s^2

        # Velocity update
        self.vel += RenderingSettings.DT * a
        self.vel += 0.1 * (a - RenderingSettings.ICE_FRICTION * self.vel)

        # Position update
        self.pos += RenderingSettings.DT * self.vel


class Target(StationaryEntity):
    def __init__(self):
        self.star_images = [self._load_image(f"assets/star_{i}.png", RenderingSettings.TARGET_WIDTH) for i in
                            range(1, 5)]
        self.respawn()
        super().__init__(*self.pos)

    @staticmethod
    def __random_position():
        margin = 50  # cm

        x = random.randint(margin, RenderingSettings.WIDTH - margin) / 100
        y = random.randint(margin, RenderingSettings.HEIGHT - margin) / 100

        return np.array([x, y], dtype=np.float32)

    def respawn(self):
        self.background = random.choice(self.star_images)
        self.pos = self.__random_position()
