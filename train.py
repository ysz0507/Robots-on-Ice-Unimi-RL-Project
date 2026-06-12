import random

import numpy as np
import tqdm

from agents.advantage_actor_critic import AdvantageActorCriticAgent
from agents.agent import Agent, Transition
from agents.scaling_wrapper import ScalingWrapper
from buffer import ReplayBuffer
from environment import IceEnv
from settings import TrainingSettings


def main():
    random.seed(TrainingSettings.SEED)
    np.random.seed(TrainingSettings.SEED)

    env = IceEnv()
    agent: Agent = ScalingWrapper(AdvantageActorCriticAgent())
    buffer = ReplayBuffer(TrainingSettings.BUFFER_SIZE)

    state = env.reset()

    for _ in tqdm.tqdm(range(TrainingSettings.EPISODES)):

        # Collect one trajectory
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.store(Transition(state, action, reward, next_state, done))
            state = next_state

        state = env.reset()

        agent.train(buffer.sample(TrainingSettings.BATCH_SIZE))


if __name__ == "__main__":
    main()
