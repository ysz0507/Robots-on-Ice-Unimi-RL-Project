from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal

from agents.agent import Agent
from settings import TrainingSettings, RenderingSettings


# ---------------------------------------------------------------------------
# Neural-network building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Generic multi-layer perceptron with ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Squashed Gaussian policy.

    Outputs a mean and log-std for each action dimension.
    Actions are sampled via the re-parameterisation trick and squashed
    through tanh so they stay within [-1, 1].
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        """Return (mean, log_std) before squashing."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Returns
        -------
        action      : tanh-squashed action in (-1, 1)
        log_prob    : log π(a|s) corrected for the tanh transformation
        mean_action : deterministic action (tanh of mean), used at eval time
        """
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Re-parameterisation trick
        x = dist.rsample()                              # pre-squash sample
        action = torch.tanh(x)

        # Log-prob with Jacobian correction for tanh squashing
        # log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)   # scalar per sample

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class TwinQNetwork(nn.Module):
    """
    Two independent Q-networks (critics) sharing the same class.
    Using two separate networks (Clipped Double-Q) reduces over-estimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Return (q1, q2) — both needed for the critic loss."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Clipped Double-Q value: min(q1, q2)."""
        q1, q2 = self(state, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent(Agent):
    """
    Soft Actor-Critic (Haarnoja et al., 2018 / 2019).

    Key design choices
    ------------------
    * Clipped Double-Q critics with a soft target network (Polyak averaging).
    * Squashed Gaussian policy with re-param trick.
    * Automatic entropy tuning: α is learnt by gradient descent on a
      temperature loss so that the policy entropy stays near a target value
      (−action_dim by default).

    Expected entries in TrainingSettings
    -------------------------------------
    state_dim       : int
    action_dim      : int
    hidden_dim      : int   (default 256)
    lr_actor        : float (default 3e-4)
    lr_critic       : float (default 3e-4)
    lr_alpha        : float (default 3e-4)
    gamma           : float (default 0.99)
    tau             : float (default 0.005)  — Polyak averaging coefficient
    target_entropy  : float (default -action_dim)
    init_alpha      : float (default 0.2)
    device          : str   (default "cpu")
    """

    def get_models(self) -> Sequence:
        return self.actor, self.critic

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        state_dim = 8 if RenderingSettings().ENABLE_METEORITE else 4
        action_dim = 2
        self.gamma = TrainingSettings().DISCOUNT_FACTOR
        self.tau = TrainingSettings().TAU
        self.device = torch.device("cuda")

        # ---- Actor -------------------------------------------------------
        self.actor = GaussianActor(state_dim, action_dim, TrainingSettings().HIDDEN_ACTOR_NODES).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=TrainingSettings().ACTOR_LEARNING_RATE
        )

        # ---- Critics (online + target) -----------------------------------
        self.critic = TwinQNetwork(state_dim, action_dim, TrainingSettings().HIDDEN_CRITIC_NODES).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, TrainingSettings().HIDDEN_CRITIC_NODES).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=TrainingSettings().CRITIC_LEARNING_RATE
        )

        # ---- Automatic entropy tuning ------------------------------------
        self.target_entropy = TrainingSettings().TARGET_ENTROPY
        init_alpha = TrainingSettings().INIT_ALPHA
        # log_alpha is the learnable parameter; alpha = exp(log_alpha) ≥ 0
        self.log_alpha = torch.tensor(
            [torch.log(torch.tensor(init_alpha))],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=TrainingSettings().ALPHA_LEARNING_RATE
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """Temperature coefficient (always positive)."""
        return self.log_alpha.exp()

    def _soft_update_target(self):
        """Polyak-average critic weights into the target network."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state):
        """
        Returns a numpy action sampled from the current policy.
        Uses the stochastic policy during training and the deterministic
        mean at evaluation time — here we default to stochastic.
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        action, _, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Core SAC update (private)
    # ------------------------------------------------------------------

    def train(
        self,
        states:      torch.Tensor,   # (B, state_dim)
        actions:     torch.Tensor,   # (B, action_dim)
        rewards:     torch.Tensor,   # (B,)
        next_states: torch.Tensor,   # (B, state_dim)
        dones:       torch.Tensor,   # (B,)
    ):
        """
        One gradient step for the critic, actor, and temperature.

        Returns
        -------
        critic_loss : float
        actor_loss  : float
        """
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards = rewards.to(self.device)  # (B, 1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)  # (B, 1)

        alpha = self.alpha.detach()   # treat as constant when updating critic

        # ------------------------------------------------------------------ #
        # 1. Critic update
        #    y = r + γ (1−d) [min_Q(s', ã') − α log π(ã'|s')]
        #    L_Q = E[(Q(s,a) − y)²]
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q_next = self.critic_target.q_min(next_states, next_actions)   # (B, 1)
            # Soft Bellman target
            target_q = rewards + self.gamma * (1.0 - dones) * (q_next - alpha * next_log_pi)

        q1, q2 = self.critic(states, actions)
        critic_loss = (
            nn.functional.mse_loss(q1, target_q)
            + nn.functional.mse_loss(q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ #
        # 2. Actor update
        #    L_π = E[α log π(ã|s) − min_Q(s, ã)]
        #    (maximise expected return + entropy)
        # ------------------------------------------------------------------ #
        new_actions, log_pi, _ = self.actor.sample(states)
        q_val = self.critic.q_min(states, new_actions)          # (B, 1)
        actor_loss = (alpha * log_pi - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------------ #
        # 3. Temperature (α) update
        #    L_α = E[−α (log π(ã|s) + H_target)]
        #    Gradient descends so α rises when entropy < target, falls otherwise.
        # ------------------------------------------------------------------ #
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------ #
        # 4. Soft-update target critic
        # ------------------------------------------------------------------ #
        self._soft_update_target()

        return critic_loss, actor_loss
