import random

import numpy as np
import pygame

from settings import Settings


class StationaryEntity:
    def __init__(self, x, y, image_path, width):
        self.pos = np.array([x, y], dtype=np.float32)
        self.background = self._load_image(image_path, width)

    @staticmethod
    def _load_image(path, width):
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
        super().__init__(x, y, "assets/robot.png", Settings.ROBOT_WIDTH)
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
        if norm > Settings.MAX_FORCE:
            force = force / norm * Settings.MAX_FORCE

        # Position update
        self.pos += Settings.DT * self.vel

        # Velocity update
        self.vel += Settings.DT * (force / self.mass)
        self.vel -= Settings.ICE_FRICTION * self.vel


class Target(StationaryEntity):
    def __init__(self):
        pos = self.__random_position()
        super().__init__(pos[0], pos[1], self.__random_star_path(), Settings.TARGET_WIDTH)

    @staticmethod
    def __random_star_path():
        index = random.randint(1, 4)
        return f"assets/star_{index}.png"

    @staticmethod
    def __random_position():
        margin = 50

        x = random.randint(margin, Settings.WIDTH - margin)
        y = random.randint(margin, Settings.HEIGHT - margin)

        return np.array([x, y], dtype=np.float32)

    def respawn(self):
        self.background = self._load_image(self.__random_star_path(), Settings.TARGET_WIDTH)
        self.pos = self.__random_position()


class IceEnv:
    def __init__(self):
        self.robot = Robot(Settings.WIDTH // 2, Settings.HEIGHT // 2, Settings.ROBOT_MASS)
        self.target = Target()

        self.step_count = 0
        self.done = False
        tile = pygame.image.load("assets/moon_tile.png").convert()
        self.background = pygame.Surface((Settings.WIDTH, Settings.HEIGHT))
        for x in range(0, Settings.WIDTH, tile.get_width()):
            for y in range(0, Settings.HEIGHT, tile.get_height()):
                self.background.blit(tile, (x, y))

    def reset(self):
        self.robot.reset(Settings.WIDTH // 2, Settings.HEIGHT // 2)
        self.target.respawn()

        self.step_count = 0
        self.done = False

        return self.get_state()

    def get_state(self):
        """
        Example state representation.
        """

        return np.concatenate([
            self.robot.pos,
            self.robot.vel,
            self.target.pos
        ])

    def compute_reward(self, force):
        """
        Reward = negative distance penalty
                 - energy penalty
        """

        distance = np.linalg.norm(
            self.robot.pos - self.target.pos
        )

        energy = (Settings.ENERGY_COEFF / 100.0) * np.linalg.norm(force) ** 2

        reward = -(distance ** 2) - energy

        return reward

    def check_target_reached(self):
        distance = np.linalg.norm(
            self.robot.pos - self.target.pos
        )

        return distance < Settings.COLLECT_DISTANCE

    def step(self, action):
        """
        Action = force vector [Fx, Fy]
        """

        self.robot.update(action)

        reward = self.compute_reward(action)

        # Target reached
        if self.check_target_reached():
            reward += 1000
            self.target.respawn()

        self.step_count += 1

        # Episode termination
        if self.step_count >= Settings.MAX_STEPS:
            self.done = True

        next_state = self.get_state()

        return next_state, reward, self.done

    def draw(self, screen):
        screen.blit(self.background, (0, 0))

        self.target.draw(screen)
        self.robot.draw(screen)
