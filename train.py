import os
import random
from datetime import datetime

import numpy as np
import pygame
import torch
import tqdm

import wandb
from agents.agent import Agent, Transition
from agents.soft_actor_critic import SACAgent
from buffer import ReplayBuffer
from environment import RecordedIceEnv, ScaledIceEnv
from settings import TrainingSettings, RenderingSettings

os.environ["WANDB_SILENT"] = "true"

def main():
    random.seed(TrainingSettings().SEED)
    np.random.seed(TrainingSettings().SEED)
    torch.manual_seed(TrainingSettings().SEED)

    wandb.init(
        project="Robots-On-Ice",
        config={
            "TrainingSettings": TrainingSettings().as_dict(),
            "RenderingSettings": RenderingSettings().as_dict(),
        },
        name=f"SAC Agent Training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    wandb.define_metric("*", step_metric="episode")

    pygame.init()
    screen = pygame.display.set_mode((RenderingSettings().WIDTH, RenderingSettings().HEIGHT), flags=pygame.HIDDEN)

    training_env = ScaledIceEnv()
    recording_env = RecordedIceEnv()
    agent: Agent = SACAgent()

    # wandb.watch(agent.get_models(), log="all", log_freq=TrainingSettings().LOG_FREQ)

    replay_buffer = ReplayBuffer(TrainingSettings().BUFFER_SIZE)

    # Fill up replay buffer
    with tqdm.tqdm(total=TrainingSettings().WARMUP_STEPS, desc="Warmup") as pbar:
        while len(replay_buffer) < TrainingSettings().WARMUP_STEPS:
            state = training_env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = training_env.step(action)
                replay_buffer.store(Transition(state, action, reward, next_state, done))
                pbar.update(1)
                state = next_state

    # Actual training
    for episode in tqdm.tqdm(range(TrainingSettings().EPISODES)):

        episode_return = 0.0
        actor_loss = critic_loss = 0.0

        # Collect one trajectory
        done = False
        is_recording = episode % TrainingSettings().VIDEO_FREQ == 0
        env = recording_env if is_recording else training_env
        state = env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.store(Transition(state, action, reward, next_state, done))
            episode_return += reward
            state = next_state

            batch = replay_buffer.sample(TrainingSettings().BATCH_SIZE)
            actor_loss, critic_loss = agent.train(*batch)

            if is_recording:
                env.draw(screen)
        if is_recording:
            video = wandb.Video(env.get_frames((0, 3, 2, 1)), format="mp4", fps=RenderingSettings().FPS)
            wandb.log({"episode": episode, "video": video, "eval/return": episode_return,
                       "eval/targets_collected": env.targets_collected}, step=episode)

        if episode % TrainingSettings().LOG_FREQ == 0:
            wandb.log(
                {
                    "episode": episode,
                    "train/return": episode_return,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "train/targets_collected": training_env.targets_collected,
                    "buffer_size": len(replay_buffer),
                }, step=episode
            )


def train():
    try:
        main()
    finally:
        wandb.finish()
        pygame.quit()


def train_with_seeds():
    for seed in (42, 43, 44):
        TrainingSettings().SEED = seed
        train()
    TrainingSettings.clear_instance()
    RenderingSettings.clear_instance()


if __name__ == "__main__":
    train()
    # for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    #     TrainingSettings().ENERGY_COEFF = c
    #     train()
    #
    # for weight in (50, 60, 70, 80, 90, 100):
    #     RenderingSettings().ROBOT_MASS = weight
    #     train()

    # Shutdown
    # subprocess.run(["shutdown", "now"])
