import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False):
        super(ActorCritic, self).__init__()
        self.continuous = continuous

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.log_std = None

        self.apply(self._init_weights)
        orthogonal_init(self.actor[-1], gain=0.01)
        orthogonal_init(self.critic[-1], gain=1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            gain = nn.init.calculate_gain('tanh')
            nn.init.orthogonal_(module.weight.data, gain=gain)
            nn.init.zeros_(module.bias.data)

    def forward(self, state):
        if self.continuous:
            mean = self.actor(state)
            log_std = self.log_std.expand_as(mean)
            value = self.critic(state)
            return mean, log_std, value
        else:
            logits = self.actor(state)
            value = self.critic(state)
            return logits, value

    def act(self, state):
        if self.continuous:
            mean, log_std, value = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.detach(), log_prob.detach(), value.detach()
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach(), log_prob.detach(), value.detach()

    def evaluate(self, state, action):
        if self.continuous:
            mean, log_std, value = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, value
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action.squeeze(-1))
            entropy = dist.entropy()
            return log_prob, entropy, value


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        continuous=False,
        lr=2.5e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=8,
        batch_size=64,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.continuous = continuous
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.start_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.start_lr = lr

        self.policy = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(state_tensor)
        if self.continuous:
            action = action.squeeze(0).cpu().numpy()
        else:
            action = action.item()
        return action, log_prob.item(), value.item()

    def update_hyperparams(self, current_episode, max_episodes):
        frac = 1.0 - float(current_episode) / float(max_episodes)
        # Keep small floors so annealing does not drive exploration or step size to zero.
        lr = self.start_lr * max(frac, 0.1)
        self.entropy_coef = self.start_entropy_coef * max(frac, 0.2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, memory, current_episode, max_episodes):
        self.update_hyperparams(current_episode, max_episodes)

        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)
        masks = torch.tensor([0.0 if done else 1.0 for done in memory.is_terminals], dtype=torch.float32, device=self.device)
        values = torch.tensor(memory.values, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        # Bootstrap non-terminal rollout fragments from the last stored value estimate.
        next_value = values[-1] if not memory.is_terminals[-1] else values.new_tensor(0.0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            next_value = values[step]

        returns = advantages + values
        # Normalize value targets to reduce critic target scale variance.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach().squeeze(-1)

        total_steps = len(old_states)
        for _ in range(self.K_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                states = old_states[batch_idx]
                actions = old_actions[batch_idx]
                old_logprobs_batch = old_logprobs[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]

                logprobs, entropy, state_values = self.policy.evaluate(states, actions)
                state_values = state_values.squeeze(-1)

                # Clip normalized advantages to prevent outliers from dominating policy updates.
                advantages_batch = torch.clamp(advantages_batch, -5, 5)

                ratios = torch.exp(logprobs - old_logprobs_batch)
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # PPO value clipping must compare against fixed old predictions.
                value_old = values[batch_idx].detach()
                value_clipped = value_old + torch.clamp(state_values - value_old, -self.eps_clip, self.eps_clip)
                value_loss = 0.5 * (F.mse_loss(state_values, returns_batch) + F.mse_loss(value_clipped, returns_batch))
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]