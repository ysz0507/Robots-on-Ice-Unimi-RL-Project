import random
from datetime import datetime

import numpy as np
import pygame
import torch
import tqdm

import wandb
from agents.agent import Agent, Transition
from agents.scaling_wrapper import ScalingWrapper
from agents.soft_actor_critic import SACAgent
from buffer import ReplayBuffer
from environment import IceEnv, RecordedIceEnv
from settings import TrainingSettings, RenderingSettings


def main():
    random.seed(TrainingSettings.SEED)
    np.random.seed(TrainingSettings.SEED)
    torch.manual_seed(TrainingSettings.SEED)

    wandb.init(
        project="Robots-On-Ice",
        config={
            "TrainingSettings": TrainingSettings.as_dict(),
            "RenderingSettings": RenderingSettings.as_dict(),
        },
        name=f"A2C Agent Training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    wandb.define_metric("*", step_metric="episode")

    pygame.init()
    screen = pygame.display.set_mode((RenderingSettings.WIDTH, RenderingSettings.HEIGHT), flags=pygame.HIDDEN)

    training_env = IceEnv()
    recording_env = RecordedIceEnv()
    agent: Agent = ScalingWrapper(SACAgent())

    wandb.watch(agent.get_models(), log="all", log_freq=TrainingSettings.LOG_FREQ)

    replay_buffer = ReplayBuffer(TrainingSettings.BUFFER_SIZE)

    for episode in tqdm.tqdm(range(TrainingSettings.EPISODES)):

        episode_return = 0.0

        # Collect one trajectory
        done = False
        is_recording = episode % TrainingSettings.VIDEO_FREQ == 0
        env = recording_env if is_recording else training_env
        state = env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.store(Transition(state, action, reward, next_state, done))
            episode_return += reward
            state = next_state
            if is_recording:
                env.draw(screen)

        if is_recording:
            video = wandb.Video(env.get_frames((0, 3, 2, 1)), format="mp4", fps=RenderingSettings.FPS)
            wandb.log({"episode": episode, "video": video, "eval_return": episode_return,
                       "eval_targets_collected": env.targets_collected})

        batch = replay_buffer.sample(TrainingSettings.BUFFER_SIZE)
        actor_loss, critic_loss = agent.train(batch)

        if episode % TrainingSettings.LOG_FREQ == 0:
            wandb.log(
                {
                    "episode": episode,
                    "episode_return": episode_return,
                    "actor_loss": float(actor_loss),
                    "critic_loss": float(critic_loss),
                    "targets_collected": training_env.targets_collected,
                }
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        wandb.finish()
        pygame.quit()
    # Shutdown
    # subprocess.run(["shutdown", "now"])
