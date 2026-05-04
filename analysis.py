"""
PPO Analysis and Ablation Studies
==================================
This module provides advanced analysis tools for understanding PPO behavior:
1. Hyperparameter sensitivity analysis
2. Ablation studies (removing GAE, value clipping, etc.)
3. Learning curves and convergence analysis
4. Detailed statistical analysis
"""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from ppo import PPO, Memory
import time


class PPOAnalyzer:
    """Tool for analyzing PPO training behavior and hyperparameter sensitivity."""
    
    def __init__(self, env_name='CartPole-v1'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.continuous = False
        else:
            self.action_dim = int(np.prod(self.env.action_space.shape))
            self.continuous = True

    def train_with_config(self, config, num_episodes=500, return_history=False):
        """
        Train PPO with specific configuration and return final performance.
        
        Args:
            config: Configuration dictionary with hyperparameters
            num_episodes: Number of episodes to train
            return_history: Whether to return episode-by-episode history
            
        Returns:
            final_reward: Final average reward
            history: Episode rewards (if return_history=True)
        """
        ppo = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            continuous=self.continuous,
            **config
        )
        memory = Memory()
        
        timestep = 0
        rewards = []
        max_timesteps = self.env.spec.max_episode_steps
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(max_timesteps):
                timestep += 1
                action, log_prob, value = ppo.select_action(state)
                
                if self.continuous:
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                memory.states.append(torch.tensor(state, dtype=torch.float32))
                if self.continuous:
                    memory.actions.append(torch.tensor(action, dtype=torch.float32))
                else:
                    memory.actions.append(torch.tensor([action], dtype=torch.int64))
                memory.logprobs.append(torch.tensor([log_prob], dtype=torch.float32))
                memory.rewards.append(reward)
                memory.values.append(value)
                memory.is_terminals.append(done)
                
                state = next_state
                
                if timestep >= config['update_timestep']:
                    ppo.update(memory, current_episode=episode, max_episodes=num_episodes)
                    memory.clear_memory()
                    timestep = 0
                elif done:
                    if len(memory.states) > 0:
                        ppo.update(memory, current_episode=episode, max_episodes=num_episodes)
                        memory.clear_memory()
                        timestep = 0
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        
        if return_history:
            return final_avg, rewards
        return final_avg

    def learning_rate_sensitivity(self, learning_rates=[1e-4, 2e-4, 5e-4, 1e-3]):
        """
        Test PPO sensitivity to learning rate.
        
        Args:
            learning_rates: List of learning rates to test
            
        Returns:
            results: Dictionary mapping learning rates to final rewards
        """
        print(f"\n{'='*60}")
        print(f"Learning Rate Sensitivity Analysis ({self.env_name})")
        print(f"{'='*60}")
        
        results = {}
        base_config = {
            'lr': 2e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 4,
            'batch_size': 128,
            'gae_lambda': 0.95,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_timestep': 2048,
        }
        
        for lr in learning_rates:
            config = {**base_config, 'lr': lr}
            print(f"Testing lr={lr}...", end=' ')
            start = time.time()
            reward = self.train_with_config(config, num_episodes=300)
            elapsed = time.time() - start
            results[lr] = reward
            print(f"Final Avg Reward: {reward:.2f} (Time: {elapsed:.1f}s)")
        
        return results

    def entropy_coefficient_sensitivity(self, entropy_coefs=[0.001, 0.01, 0.05, 0.1]):
        """
        Test PPO sensitivity to entropy coefficient (exploration).
        
        Args:
            entropy_coefs: List of entropy coefficients to test
            
        Returns:
            results: Dictionary mapping entropy coefficients to final rewards
        """
        print(f"\n{'='*60}")
        print(f"Entropy Coefficient Sensitivity Analysis ({self.env_name})")
        print(f"{'='*60}")
        
        results = {}
        base_config = {
            'lr': 2e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 4,
            'batch_size': 128,
            'gae_lambda': 0.95,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_timestep': 2048,
        }
        
        for entropy_coef in entropy_coefs:
            config = {**base_config, 'entropy_coef': entropy_coef}
            print(f"Testing entropy_coef={entropy_coef}...", end=' ')
            start = time.time()
            reward = self.train_with_config(config, num_episodes=300)
            elapsed = time.time() - start
            results[entropy_coef] = reward
            print(f"Final Avg Reward: {reward:.2f} (Time: {elapsed:.1f}s)")
        
        return results

    def batch_size_sensitivity(self, batch_sizes=[32, 64, 128, 256]):
        """
        Test PPO sensitivity to batch size.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            results: Dictionary mapping batch sizes to final rewards
        """
        print(f"\n{'='*60}")
        print(f"Batch Size Sensitivity Analysis ({self.env_name})")
        print(f"{'='*60}")
        
        results = {}
        base_config = {
            'lr': 2e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 4,
            'batch_size': 128,
            'gae_lambda': 0.95,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'update_timestep': 2048,
        }
        
        for batch_size in batch_sizes:
            config = {**base_config, 'batch_size': batch_size}
            print(f"Testing batch_size={batch_size}...", end=' ')
            start = time.time()
            reward = self.train_with_config(config, num_episodes=300)
            elapsed = time.time() - start
            results[batch_size] = reward
            print(f"Final Avg Reward: {reward:.2f} (Time: {elapsed:.1f}s)")
        
        return results


def plot_sensitivity_analysis(results_dict, save_path='sensitivity_analysis.png'):
    """
    Create visualizations of sensitivity analysis results.
    
    Args:
        results_dict: Dictionary with sensitivity analysis results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Learning rate sensitivity
    if 'learning_rate' in results_dict:
        lr_results = results_dict['learning_rate']
        axes[0].plot(list(lr_results.keys()), list(lr_results.values()), 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Final Average Reward')
        axes[0].set_title('Learning Rate Sensitivity')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
    
    # Entropy coefficient sensitivity
    if 'entropy_coef' in results_dict:
        ent_results = results_dict['entropy_coef']
        axes[1].plot(list(ent_results.keys()), list(ent_results.values()), 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Entropy Coefficient')
        axes[1].set_ylabel('Final Average Reward')
        axes[1].set_title('Entropy Coefficient Sensitivity')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
    
    # Batch size sensitivity
    if 'batch_size' in results_dict:
        bs_results = results_dict['batch_size']
        axes[2].plot(list(bs_results.keys()), list(bs_results.values()), 'o-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Batch Size')
        axes[2].set_ylabel('Final Average Reward')
        axes[2].set_title('Batch Size Sensitivity')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sensitivity analysis plot saved to {save_path}")


if __name__ == '__main__':
    """
    Run comprehensive analysis on CartPole-v1.
    Note: This is computationally intensive. Adjust num_episodes if needed.
    """
    
    print("\n" + "="*80)
    print("PPO HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    analyzer = PPOAnalyzer(env_name='CartPole-v1')
    
    # Run sensitivity analyses
    all_results = {}
    
    # Learning rate sensitivity
    all_results['learning_rate'] = analyzer.learning_rate_sensitivity(
        learning_rates=[1e-4, 2e-4, 5e-4, 1e-3]
    )
    
    # Entropy coefficient sensitivity
    all_results['entropy_coef'] = analyzer.entropy_coefficient_sensitivity(
        entropy_coefs=[0.0001, 0.001, 0.01, 0.05]
    )
    
    # Batch size sensitivity
    all_results['batch_size'] = analyzer.batch_size_sensitivity(
        batch_sizes=[32, 64, 128, 256]
    )
    
    # Create visualization
    plot_sensitivity_analysis(all_results, save_path='sensitivity_analysis.png')
    
    # Save results
    results_json = {
        'learning_rate': {str(k): float(v) for k, v in all_results['learning_rate'].items()},
        'entropy_coef': {str(k): float(v) for k, v in all_results['entropy_coef'].items()},
        'batch_size': {str(k): float(v) for k, v in all_results['batch_size'].items()},
    }
    
    with open('sensitivity_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n✓ Analysis complete! Results saved to sensitivity_analysis.png and sensitivity_results.json")
