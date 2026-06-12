import random

import numpy as np
import pygame

from agents.human_agent import HumanAgent
from settings import RenderingSettings, TrainingSettings


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


class IceEnv:
    def __init__(self):
        self.robot = Robot(RenderingSettings.WIDTH // 2, RenderingSettings.HEIGHT // 2, RenderingSettings.ROBOT_MASS)
        self.target = Target()

        self.step_count = 0
        self.done = False

        if pygame.get_init():
            tile = pygame.image.load("assets/moon_tile.png").convert()
            self.background = pygame.Surface((RenderingSettings.WIDTH, RenderingSettings.HEIGHT))
            for x in range(0, RenderingSettings.WIDTH, tile.get_width()):
                for y in range(0, RenderingSettings.HEIGHT, tile.get_height()):
                    self.background.blit(tile, (x, y))

    def reset(self):
        self.robot.reset(RenderingSettings.WIDTH // 2, RenderingSettings.HEIGHT // 2)
        self.target.respawn()

        self.step_count = 0
        self.done = False

        return self.get_state()

    def get_state(self):
        """
        Example state representation.
        """

        return np.concatenate([
            self.target.pos - self.robot.pos,
            self.robot.vel,
        ])

    def compute_reward(self, force):
        def compute_distance_penalty():
            distance = np.linalg.norm(
                self.robot.pos - self.target.pos
            )
            return -distance ** 2

        def compute_energy_penalty():
            energy = np.linalg.norm(force)
            return -TrainingSettings.ENERGY_COEFF * energy ** 2

        return compute_distance_penalty() + compute_energy_penalty()

    def check_target_reached(self):
        distance = np.linalg.norm(
            self.robot.pos - self.target.pos
        )

        return distance < RenderingSettings.COLLECT_DISTANCE

    def step(self, action):
        """
        Action = force vector [Fx, Fy]
        """

        self.robot.update(action)

        reward = self.compute_reward(action)

        # Target reached
        if self.check_target_reached():
            reward += 1000
            self.step_count = 0
            self.target.respawn()

        self.step_count += 1

        # Episode termination
        if self.step_count >= RenderingSettings.MAX_STEPS:
            self.done = True

        next_state = self.get_state()

        return next_state, reward, self.done

    def draw(self, screen):
        screen.blit(self.background, (0, 0))

        self.target.draw(screen)
        self.robot.draw(screen)


def main():
    pygame.init()

    screen = pygame.display.set_mode((RenderingSettings.WIDTH, RenderingSettings.HEIGHT))
    pygame.display.set_caption("Robots on Ice")

    clock = pygame.time.Clock()

    env = IceEnv()
    agent = HumanAgent()

    state = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                running = False
        action = agent.select_action(state)
        _, _, done = env.step(action)

        env.draw(screen)
        pygame.display.flip()

        if done:
            print("Episode finished")
            state = env.reset()

        clock.tick(RenderingSettings.FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
