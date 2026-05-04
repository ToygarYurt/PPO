"""
Proximal Policy Optimization (PPO) Implementation
================================================
This module implements the PPO algorithm with both discrete and continuous action spaces.
PPO is a state-of-the-art policy gradient algorithm that balances sample efficiency and 
training stability through policy clipping and generalized advantage estimation (GAE).

Key Features:
- Orthogonal weight initialization for better convergence
- Generalized Advantage Estimation (GAE) for lower variance advantage estimates
- Entropy regularization for exploration
- Learning rate and entropy coefficient annealing
- Value function clipping for stable critic training
- Gradient clipping for additional stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(layer, gain=1.0):
    """
    Initialize layer weights using orthogonal initialization.
    
    Orthogonal initialization helps with training stability and convergence speed,
    especially important for deep networks in RL.
    
    Args:
        layer: Neural network layer to initialize
        gain: Scaling factor for initialization (default: 1.0)
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    
    The actor outputs either action logits (discrete) or mean/std (continuous).
    The critic outputs a single value estimate for the state.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        continuous: Whether the action space is continuous (default: False for discrete)
    """
    
    def __init__(self, state_dim, action_dim, continuous=False):
        super(ActorCritic, self).__init__()
        self.continuous = continuous

        # Actor network: outputs action distribution parameters
        # Two hidden layers of 256 units with Tanh activation
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

        # Critic network: outputs value estimate
        # Same architecture as actor for consistency
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # For continuous actions, we also learn the log of standard deviation
        # Initialize to a reasonable value (-0.5) instead of 0 to have initial exploration
        # Use bounds to prevent numerical instabilities
        if self.continuous:
            self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
            self.log_std_min = -20.0  # Minimum log_std to prevent std → 0
            self.log_std_max = 2.0    # Maximum log_std to prevent exploding std
        else:
            self.log_std = None
            self.log_std_min = None
            self.log_std_max = None

        # Initialize all weights using orthogonal initialization
        self.apply(self._init_weights)
        
        # Use smaller gain for actor output (exploration) and standard gain for critic
        orthogonal_init(self.actor[-1], gain=0.01)
        orthogonal_init(self.critic[-1], gain=1.0)

    def _init_weights(self, module):
        """Apply orthogonal initialization to all Linear layers."""
        if isinstance(module, nn.Linear):
            gain = nn.init.calculate_gain('tanh')
            nn.init.orthogonal_(module.weight.data, gain=gain)
            nn.init.zeros_(module.bias.data)

    def forward(self, state):
        """
        Forward pass through actor and critic.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            For continuous: (mean, log_std, value)
            For discrete: (logits, value)
        """
        if self.continuous:
            mean = self.actor(state)
            # Clamp log_std to prevent numerical instabilities
            log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            log_std = log_std.expand_as(mean)
            value = self.critic(state)
            return mean, log_std, value
        else:
            logits = self.actor(state)
            value = self.critic(state)
            return logits, value

    def act(self, state, deterministic=False):
        """
        Sample an action from the current policy.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: Value estimate for the state
        """
        if self.continuous:
            mean, log_std, value = self.forward(state)
            std = torch.clamp(log_std.exp(), min=1e-4)  # Prevent std from being too small
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()
            # Sum log probabilities across action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.detach(), log_prob.detach(), value.detach()
        else:
            logits, value = self.forward(state)
            # Create a categorical distribution and sample
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach(), log_prob.detach(), value.detach()

    def evaluate(self, state, action):
        """
        Evaluate log probability and entropy of given actions.
        Used during training to compute policy and value losses.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim or 1]
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the distribution
            value: Value estimate for states
        """
        if self.continuous:
            mean, log_std, value = self.forward(state)
            std = torch.clamp(log_std.exp(), min=1e-4)  # Prevent std from being too small
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
    """
    Proximal Policy Optimization (PPO) Algorithm Implementation.
    
    PPO is a policy gradient method that uses clipped surrogate objective to 
    constrain policy updates, balancing between sample efficiency and stability.
    
    Key algorithmic components:
    1. Generalized Advantage Estimation (GAE) for advantage computation
    2. Clipped surrogate objective for policy updates
    3. Value function clipping for critic stability
    4. Entropy regularization for exploration
    5. Learning rate and entropy annealing over training
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        continuous: Whether action space is continuous (default: False)
        lr: Learning rate (default: 2.5e-4)
        gamma: Discount factor (default: 0.99)
        eps_clip: Clipping parameter for policy loss (default: 0.2)
        K_epochs: Number of update epochs per batch (default: 8)
        batch_size: Batch size for mini-batch updates (default: 64)
        gae_lambda: GAE lambda parameter (default: 0.95)
        value_coef: Coefficient for value loss (default: 0.5)
        entropy_coef: Coefficient for entropy loss (default: 0.01)
    """
    
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
        target_kl=None,
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
        self.target_kl = target_kl

        # Create actor-critic networks
        self.policy = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        # Keep a copy of the old policy for importance sampling
        self.policy_old = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Adam optimizer for policy updates
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state, deterministic=False):
        """
        Select an action using the current policy (old policy during collection).
        
        Args:
            state: Current state (numpy array or similar)
            
        Returns:
            action: Action to execute
            log_prob: Log probability of action
            value: Value estimate for state
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(state_tensor, deterministic=deterministic)
        if self.continuous:
            action = action.squeeze(0).cpu().numpy()
        else:
            action = action.item()
        return action, log_prob.item(), value.item()

    def get_value(self, state):
        """Return V(s) from the old policy, used to bootstrap unfinished rollouts."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.continuous:
                _, _, value = self.policy_old.forward(state_tensor)
            else:
                _, value = self.policy_old.forward(state_tensor)
        return value.item()

    def update_hyperparams(self, current_episode, max_episodes):
        """
        Anneal hyperparameters (learning rate and entropy coefficient) during training.
        
        This helps stabilize training by reducing exploration and step size as the 
        agent learns. We keep floors to prevent them from going to zero.
        
        Args:
            current_episode: Current episode number
            max_episodes: Total number of episodes
        """
        # Linear annealing: frac goes from 1.0 to 0.0
        frac = 1.0 - float(current_episode) / float(max_episodes)
        
        # Keep floors so annealing doesn't drive values to zero
        lr = self.start_lr * max(frac, 0.1)
        self.entropy_coef = self.start_entropy_coef * max(frac, 0.2)
        
        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, memory, current_episode, max_episodes, next_value=0.0):
        """
        Perform a PPO update on the collected trajectory data.
        
        Steps:
        1. Compute advantages using GAE
        2. Normalize returns and advantages
        3. Perform K_epochs of mini-batch updates with clipped surrogate objective
        4. Update old policy to match new policy
        
        Args:
            memory: Memory object containing collected trajectory data
            current_episode: Current episode for hyperparameter annealing
            max_episodes: Total episodes for annealing schedule
        """
        # Anneal learning rate and entropy coefficient
        self.update_hyperparams(current_episode, max_episodes)

        # Convert memory to tensors on device
        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)
        # Create masks: 0 if terminal, 1 if not (for GAE computation)
        masks = torch.tensor(
            [0.0 if done else 1.0 for done in memory.is_terminals], 
            dtype=torch.float32, 
            device=self.device
        )
        values = torch.tensor(memory.values, dtype=torch.float32, device=self.device)

        # ===== COMPUTE ADVANTAGES USING GAE =====
        # Generalized Advantage Estimation provides lower-variance advantage estimates
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        # Bootstrap unfinished rollout fragments with V(s_{t+1}). For terminal states
        # this is exactly 0, but for fixed-size rollouts it avoids an off-by-one target.
        next_value = values.new_tensor(next_value)
        
        for step in reversed(range(len(rewards))):
            # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            # GAE update with exponential weighting
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            next_value = values[step]

        # Compute returns as advantages + value estimates
        returns = advantages + values
        
        # Normalize only advantages. Value targets must stay on the reward scale;
        # normalizing returns makes the critic learn a moving artificial target.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Prepare old trajectories for batch updates
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach().squeeze(-1)

        # ===== PPO UPDATE EPOCHS =====
        total_steps = len(old_states)
        stop_update = False
        for epoch in range(self.K_epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(total_steps, device=self.device)
            
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Get batch data
                states = old_states[batch_idx]
                actions = old_actions[batch_idx]
                old_logprobs_batch = old_logprobs[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]

                # Evaluate current policy on batch
                logprobs, entropy, state_values = self.policy.evaluate(states, actions)
                state_values = state_values.squeeze(-1)

                # Clip advantages to prevent outliers from dominating updates
                # This is an additional stability measure
                advantages_batch = torch.clamp(advantages_batch, -5, 5)

                # ===== POLICY LOSS (PPO Clipped Surrogate Objective) =====
                # Importance sampling ratio: pi_new / pi_old
                ratios = torch.exp(logprobs - old_logprobs_batch)
                approx_kl = (old_logprobs_batch - logprobs).mean().detach()
                
                # Unclipped surrogate: ratio * advantage
                surr1 = ratios * advantages_batch
                # Clipped surrogate: prevents ratio from deviating too far from 1.0
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch
                
                # Take minimum to enforce the clipping
                policy_loss = -torch.min(surr1, surr2).mean()

                # ===== VALUE LOSS (with clipping) =====
                # PPO value clipping compares against the old value estimate
                value_old = values[batch_idx].detach()
                # Clip value prediction to not deviate too much from old estimate
                value_clipped = value_old + torch.clamp(
                    state_values - value_old, 
                    -self.eps_clip, 
                    self.eps_clip
                )
                # Take minimum MSE loss between clipped and unclipped
                value_loss = 0.5 * (
                    F.mse_loss(state_values, returns_batch) + 
                    F.mse_loss(value_clipped, returns_batch)
                )

                # ===== ENTROPY LOSS =====
                # Negative entropy to maximize entropy (encourage exploration)
                entropy_loss = -entropy.mean()

                # ===== TOTAL LOSS =====
                loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )

                # ===== BACKPROPAGATION AND OPTIMIZATION =====
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        # Update old policy to match new policy for next iteration
        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    """
    Simple buffer to store trajectories during rollout collection.
    
    Stores states, actions, log probabilities, rewards, terminal flags, and values
    collected during environment interaction.
    """
    
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        """Clear all stored data."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
