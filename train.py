import random
from datetime import datetime

import numpy as np
import tqdm

import wandb
from agents.advantage_actor_critic import AdvantageActorCriticAgent
from agents.agent import Agent, Transition
from agents.scaling_wrapper import ScalingWrapper
from buffer import ReplayBuffer
from environment import IceEnv
from settings import TrainingSettings, RenderingSettings


def main():
    random.seed(TrainingSettings.SEED)
    np.random.seed(TrainingSettings.SEED)

    wandb.init(
        project="Robots-On-Ice",
        config={
            "TrainingSettings": TrainingSettings.as_dict(),
            "RenderingSettings": RenderingSettings.as_dict(),
        },
        name=f"A2C Agent Training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    env = IceEnv()
    agent: Agent = ScalingWrapper(AdvantageActorCriticAgent())
    buffer = ReplayBuffer(TrainingSettings.BUFFER_SIZE)

    wandb.watch(agent.get_models(), log="all", log_freq=TrainingSettings.LOG_FREQ)

    state = env.reset()

    for episode in tqdm.tqdm(range(TrainingSettings.EPISODES)):

        episode_reward = 0.0

        # Collect one trajectory
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            buffer.store(
                Transition(state, action, reward, next_state, done)
            )

            episode_reward += reward
            state = next_state

        state = env.reset()

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