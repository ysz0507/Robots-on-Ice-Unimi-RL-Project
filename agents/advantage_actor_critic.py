from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import relu

from agents.agent import Transition, Agent
from settings import TrainingSettings


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)

        self.mean = nn.Linear(128, action_dim)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = relu(self.fc1(x))

        mean = self.mean(x)
        std = torch.exp(self.log_std)

        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class AdvantageActorCriticAgent(Agent):

    def get_models(self) -> Sequence:
        return self.actor, self.critic

    def __init__(self, state_dim=4, action_dim=2):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=TrainingSettings.ACTOR_LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=TrainingSettings.CRITIC_LEARNING_RATE,
        )

        self.gamma = TrainingSettings.DISCOUNT_FACTOR

    @torch.no_grad()
    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        mean, std = self.actor(state)

        dist = Normal(mean, std)
        action = dist.sample()

        return action.detach().numpy()

    def train(self, transitions: list[Transition]) -> tuple[float, float]:
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        for t in transitions:
            actor_loss, critic_loss = self.__train_single_transition(
                state=torch.tensor(t.state),
                action=torch.tensor(t.action),
                reward=t.reward,
                next_state=torch.tensor(t.next_state),
                done=t.done
            )
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss

        return total_actor_loss / len(transitions), total_critic_loss / len(transitions)

    def __train_single_transition(self, state, action, reward, next_state, done):
        value = self.critic(state)
        with torch.no_grad():
            next_value = self.critic(next_state)

        advantage = reward + (1 - done) * self.gamma * next_value - value

        # Update actor
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        # Omitting 1/(1-gamma) factor since it doesn't affect the optimization
        log_prob = dist.log_prob(action).sum()  # TODO Verify that this is correct for multi-dimensional actions
        actor_loss = -(log_prob * advantage.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        # Equivalent to w←w+βδ∇V(St)
        critic_loss = 0.5 * advantage.pow(2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
