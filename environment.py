from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import pygame

from agents.human_agent import HumanAgent
from entities import Robot, Target, Meteorite
from settings import RenderingSettings, TrainingSettings


class IceEnv:
    def __init__(self):
        self.robot = Robot(RenderingSettings.WIDTH // 200, RenderingSettings.HEIGHT // 200,
                           RenderingSettings.ROBOT_MASS)
        self.target = Target()

        self.meteorite = Meteorite() if RenderingSettings.ENABLE_METEORITE else None

        self.step_count = 0
        self.total_step_count = 0
        self.done = False
        self.targets_collected = 0

        if pygame.get_init():
            self.font = pygame.font.SysFont("Comic Sans MS", 30)
            tile = pygame.image.load("assets/moon_tile.png").convert()
            self.background = pygame.Surface((RenderingSettings.WIDTH, RenderingSettings.HEIGHT))
            for x in range(0, RenderingSettings.WIDTH, tile.get_width()):
                for y in range(0, RenderingSettings.HEIGHT, tile.get_height()):
                    self.background.blit(tile, (x, y))

    def reset(self):
        self.robot.reset(RenderingSettings.WIDTH // 200, RenderingSettings.HEIGHT // 200)
        self.target.respawn()
        if self.meteorite is not None:
            self.meteorite.respawn()

        self.step_count = 0
        self.total_step_count = 0
        self.targets_collected = 0
        self.done = False

        return self.get_state()

    def get_targets_collected(self):
        return self.targets_collected

    def get_state(self):
        """
        Example state representation.
        """
        state = [
            self.target.pos - self.robot.pos,
            self.robot.vel,
        ]
        if self.meteorite is not None:
            state.extend([
                self.meteorite.pos - self.robot.pos,
                self.meteorite.vel
            ])
        return np.concatenate(state)

    def compute_reward(self, force):
        def compute_distance_reward():
            distance = np.linalg.norm(
                self.robot.pos - self.target.pos
            )
            return -distance ** 2

        def compute_energy_penalty():
            energy = np.linalg.norm(force)
            return -TrainingSettings.ENERGY_COEFF / 100 * energy ** 2

        return compute_distance_reward() + compute_energy_penalty()

    def check_collision(self, entity, collider):
        distance = np.linalg.norm(
            self.robot.pos - entity.pos
        )
        return distance < collider / 100

    def step(self, action):
        """
        Action = force vector [Fx, Fy]
        """

        self.robot.update(action)
        if self.meteorite is not None:
            self.meteorite.update()

        reward = self.compute_reward(action)

        # Target reached
        if self.check_collision(self.target, RenderingSettings.COLLECT_DISTANCE):
            reward += TrainingSettings.COLLECTED_REWARD
            self.step_count = 0
            self.targets_collected += 1
            self.target.respawn()

        self.step_count += 1
        self.total_step_count += 1

        # Episode termination
        if self.step_count >= RenderingSettings.MAX_STEPS_PER_TARGET or self.total_step_count >= RenderingSettings.MAX_STEPS_PER_EPISODE:
            self.done = True

        meteorite_coll_distance = (RenderingSettings.METEORITE_WIDTH + RenderingSettings.ROBOT_WIDTH) / 2
        if self.meteorite is not None and self.check_collision(self.meteorite, meteorite_coll_distance):
            reward -= TrainingSettings.METEORITE_REWARD
            self.done = True

        next_state = self.get_state()

        return next_state, reward, self.done

    def __draw_text(self, string, screen, position, color=(255, 255, 255)):
        text = self.font.render(string, True, color)
        screen.blit(text, position)

    def draw(self, screen):
        screen.blit(self.background, (0, 0))
        self.target.draw(screen)
        self.robot.draw(screen)
        if self.meteorite is not None:
            self.meteorite.draw(screen)

        self.__draw_text(f"Targets collected: {self.targets_collected}", screen, (10, 10))

        time_left = RenderingSettings.DT * (RenderingSettings.MAX_STEPS_PER_EPISODE - self.total_step_count)
        self.__draw_text(f"Time left: {time_left:.2f}s", screen, (10, 50))


class ScaledIceEnv(IceEnv):
    def __init__(self):
        super().__init__()
        self.max_action = np.array([RenderingSettings.MAX_FORCE, RenderingSettings.MAX_FORCE], dtype=np.float32)
        max_state = [RenderingSettings.WIDTH / 100, RenderingSettings.HEIGHT / 100, 10, 10]
        if self.meteorite is not None:
            max_state += [RenderingSettings.WIDTH / 100, RenderingSettings.HEIGHT / 100,
                          RenderingSettings.MAX_METEORITE_SPEED, RenderingSettings.MAX_METEORITE_SPEED]
        self.max_state = np.array(max_state, dtype=np.float32)

    def __normalize_state(self, state):
        return state / self.max_state

    def __denormalize(self, action):
        return action * self.max_action

    def get_state(self):
        state = super().get_state()
        return self.__normalize_state(state)

    def step(self, action):
        denormalized_action = self.__denormalize(action)
        return super().step(denormalized_action)

class RecordedIceEnv(ScaledIceEnv):
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

    running = True

    while running:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                running = False
        action = agent.select_action(None)
        state, reward, done = env.step(action)
        print(state)

        env.draw(screen)
        pygame.display.flip()

        if done:
            print("Episode finished")
            env.reset()

        clock.tick(RenderingSettings.FPS)

    pygame.quit()
    env.save_recording()


if __name__ == "__main__":
    main()
