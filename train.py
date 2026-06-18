import random
import subprocess
from datetime import datetime

import numpy as np
import pygame
import torch
import tqdm

import wandb
from agents.advantage_actor_critic import AdvantageActorCriticAgent
from agents.agent import Agent, Transition
from agents.scaling_wrapper import ScalingWrapper
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
    agent: Agent = ScalingWrapper(AdvantageActorCriticAgent())

    wandb.watch(agent.get_models(), log="all", log_freq=TrainingSettings.LOG_FREQ)


    for episode in tqdm.tqdm(range(TrainingSettings.EPISODES)):

        episode_return = 0.0

        # Collect one trajectory
        done = False
        state = training_env.reset()
        transitions = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = training_env.step(action)
            transitions.append(Transition(state, action, reward, next_state, done))
            episode_return += reward
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
            wandb.log({"episode": episode, "video": video, "eval_return": eval_return,
                       "eval_targets_collected": recording_env.targets_collected})

        actor_loss, critic_loss = agent.train(transitions)

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
