from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import pygame

from agents.human_agent import HumanAgent
from entities import Robot, Target
from settings import RenderingSettings, TrainingSettings


class IceEnv:
    def __init__(self):
        self.robot = Robot(RenderingSettings.WIDTH // 2, RenderingSettings.HEIGHT // 2, RenderingSettings.ROBOT_MASS)
        self.target = Target()

        self.step_count = 0
        self.done = False
        self.targets_collected = 0

        self.distance_normalizer = np.linalg.norm([RenderingSettings.WIDTH, RenderingSettings.HEIGHT]) * 0.8

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
        self.targets_collected = 0
        self.done = False

        return self.get_state()

    def get_targets_collected(self):
        return self.targets_collected

    def get_state(self):
        """
        Example state representation.
        """

        return np.concatenate([
            self.target.pos - self.robot.pos,
            self.robot.vel,
        ])

    def compute_reward(self, force):
        def compute_distance_reward():
            distance = np.linalg.norm(
                self.robot.pos - self.target.pos
            ) / self.distance_normalizer
            return 0.3 / (distance + 0.25) - 0.2

        def compute_energy_penalty():
            energy = np.linalg.norm(force)
            return -TrainingSettings.ENERGY_COEFF * energy ** 2

        return compute_distance_reward()

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
            reward += 200  # Large positive reward for reaching the target
            self.step_count = 0
            self.targets_collected += 1
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


class RecordedIceEnv(IceEnv):
    def __init__(self, recording_dir: str = f"/tmp/robots_on_ice/"):
        super().__init__()
        self.recording_dir = Path(recording_dir)
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []

    def draw(self, screen):
        super().draw(screen)
        self.frames.append(pygame.surfarray.array3d(screen))

    def save_recording(self) -> Path:
        video_path = self.recording_dir / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".mp4")
        # noinspection PyTypeChecker
        imageio.mimsave(video_path, self.get_frames((0, 2, 1, 3)), fps=RenderingSettings.FPS)
        print(f"Recording saved to {video_path}")
        self.frames = []
        return video_path

    def get_frames(self, order: tuple) -> np.ndarray:
        frames = np.transpose(self.frames, order)
        self.frames = []
        return frames


def main():
    pygame.init()
    screen = pygame.display.set_mode((RenderingSettings.WIDTH, RenderingSettings.HEIGHT))
    pygame.display.set_caption("Robots on Ice")

    clock = pygame.time.Clock()

    env = RecordedIceEnv()
    agent = HumanAgent()

    state = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                running = False
        action = agent.select_action(state)
        _, reward, done = env.step(action)
        print(reward)

        env.draw(screen)
        pygame.display.flip()

        if done:
            print("Episode finished")
            state = env.reset()

        clock.tick(RenderingSettings.FPS)

    pygame.quit()
    env.save_recording()


if __name__ == "__main__":
    main()
