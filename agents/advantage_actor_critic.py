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

    def __init__(self, state_dim=2, action_dim=2):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=TrainingSettings.ACTOR_LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=TrainingSettings.CRITIC_LEARNING_RATE,
        )

        self.gamma = TrainingSettings.DISCOUNT_FACTOR

    @torch.no_grad()
    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        mean, std = self.policy_net(state)

        dist = Normal(mean, std)
        action = dist.sample()

        return action.detach().numpy()

    def train(self, transitions: list[Transition]):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            next_states.append(t.next_state)
            dones.append(t.done)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # ---- Critic targets ----

        values = self.value_net(states)

        with torch.no_grad():
            next_values = self.value_net(next_states)

            targets = rewards + (
                    self.gamma
                    * next_values
                    * (1.0 - dones)
            )

        advantages = targets - values

        # ---- Actor loss ----

        mean, std = self.policy_net(states)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)

        actor_loss = -(log_probs * advantages.detach()).mean()

        # ---- Critic loss ----

        critic_loss = advantages.pow(2).mean()

        # ---- Update actor ----

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Update critic ----

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }
