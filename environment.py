import random

import numpy as np
import pygame

from settings import Settings


class Robot:
    def __init__(self, x, y, mass):
        self.mass = mass

        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, x, y):
        self.pos[:] = [x, y]
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

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            Settings.BLUE,
            self.pos.astype(int),
            Settings.ROBOT_RADIUS
        )


class Target:
    def __init__(self):
        self.pos = self.random_position()

    def random_position(self):
        margin = 50

        x = random.randint(margin, Settings.WIDTH - margin)
        y = random.randint(margin, Settings.HEIGHT - margin)

        return np.array([x, y], dtype=np.float32)

    def respawn(self):
        self.pos = self.random_position()

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            Settings.RED,
            self.pos.astype(int),
            Settings.TARGET_RADIUS
        )


class IceEnv:
    def __init__(self):
        self.robot = Robot(Settings.WIDTH // 2, Settings.HEIGHT // 2, Settings.ROBOT_MASS)
        self.target = Target()

        self.step_count = 0
        self.done = False

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

        return distance < 25

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
        screen.fill(Settings.WHITE)

        self.target.draw(screen)
        self.robot.draw(screen)
