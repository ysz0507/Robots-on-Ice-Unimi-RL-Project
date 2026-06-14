import abc

import numpy as np

from agents.agent import Agent, Transition
from settings import RenderingSettings


class ScalingWrapper(Agent, metaclass=abc.ABCMeta):
    def __init__(self, model: Agent):
        self.model = model
        self.max_action = np.array([RenderingSettings.MAX_FORCE, RenderingSettings.MAX_FORCE], dtype=np.float32)
        self.max_state = np.array([RenderingSettings.WIDTH, RenderingSettings.HEIGHT, 100, 100], dtype=np.float32)

    def __normalize_state(self, state):
        return state / self.max_state

    def __normalize_action(self, action):
        return action / self.max_action

    def select_action(self, state):
        normalized_state = self.__normalize_state(state)
        normalized_action = self.model.select_action(normalized_state)
        return normalized_action * self.max_action

    def get_models(self):
        return self.model.get_models()

    def train(self, transitions: list[Transition]) -> tuple[float, float]:
        for i in range(len(transitions)):
            transitions[i] = Transition(
                state=self.__normalize_state(transitions[i].state),
                action=self.__normalize_action(transitions[i].action),
                reward=transitions[i].reward,
                next_state=self.__normalize_state(transitions[i].next_state),
                done=transitions[i].done
            )
        return self.model.train(transitions)
