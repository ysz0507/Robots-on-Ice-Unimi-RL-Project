import random
from datetime import datetime

import numpy as np
import pygame
import torch
import tqdm

import wandb
from agents.advantage_actor_critic import AdvantageActorCriticAgent
from agents.agent import Agent, Transition
from agents.scaling_wrapper import ScalingWrapper
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

    pygame.init()
    screen = pygame.display.set_mode((RenderingSettings.WIDTH, RenderingSettings.HEIGHT), flags=pygame.HIDDEN)

    training_env = IceEnv()
    recording_env = RecordedIceEnv()
    agent: Agent = ScalingWrapper(AdvantageActorCriticAgent())
    buffer = ReplayBuffer(TrainingSettings.BUFFER_SIZE)

    wandb.watch(agent.get_models(), log="all", log_freq=TrainingSettings.LOG_FREQ)


    for episode in tqdm.tqdm(range(TrainingSettings.EPISODES)):

        episode_reward = 0.0

        # Collect one trajectory
        done = False

        state = training_env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = training_env.step(action)

            buffer.store(
                Transition(state, action, reward, next_state, done)
            )

            episode_reward += reward
            state = next_state

        if episode % TrainingSettings.VIDEO_FREQ == 0:
            recording_env.reset()
            done = False
            eval_return = 0.0
            while not done:
                action = agent.select_action(recording_env.get_state())
                _, reward, done = recording_env.step(action)
                eval_return += reward
                recording_env.draw(screen)
            video = wandb.Video(recording_env.get_frames((0, 3, 2, 1)), format="mp4", fps=RenderingSettings.FPS)
            wandb.log({"episode": episode, "video": video, "eval_return": eval_return})

        actor_loss, critic_loss = agent.train(
            buffer.sample(TrainingSettings.BATCH_SIZE)
        )

        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "buffer_size": len(buffer),
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()