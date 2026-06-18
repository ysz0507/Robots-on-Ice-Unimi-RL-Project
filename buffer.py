import numpy as np
import torch

from agents.agent import Transition


class ReplayBuffer:
    def __init__(self, capacity):
        state_dim = 4
        action_dim = 2
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, transition: Transition):
        self.states[self.position] = transition.state
        self.actions[self.position] = transition.action
        self.rewards[self.position] = transition.reward
        self.next_states[self.position] = transition.next_state
        self.dones[self.position] = transition.done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size
