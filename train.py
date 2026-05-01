import gymnasium as gym
import torch
import numpy as np
from ppo import PPO, Memory
import matplotlib.pyplot as plt


env_configs = {
    'CartPole-v1': {
        'max_episodes': 500,
        'max_timesteps': 500,
        'update_timestep': 2048,
        'lr': 3e-4,
        'K_epochs': 4,
        'batch_size': 64,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'eps_clip': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'solved_reward': 195,
    },
    'LunarLander-v3': {
        'max_episodes': 3000,
        'max_timesteps': 1000,
        'update_timestep': 8192,
        'lr': 2e-4,
        'K_epochs': 5,
        'batch_size': 128,
        'gamma': 0.99,
        'gae_lambda': 0.98,
        'eps_clip': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'solved_reward': 200,
    },
}


def train(env_name):
    config = env_configs[env_name]
    env = gym.make(env_name)
    state_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = int(np.prod(env.action_space.shape))
        continuous = True

    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        lr=config['lr'],
        gamma=config['gamma'],
        eps_clip=config['eps_clip'],
        K_epochs=config['K_epochs'],
        batch_size=config['batch_size'],
        gae_lambda=config['gae_lambda'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
    )
    memory = Memory()

    timestep = 0
    rewards = []
    max_timesteps = env.spec.max_episode_steps if getattr(env, 'spec', None) and env.spec.max_episode_steps is not None else config['max_timesteps']

    for episode in range(config['max_episodes']):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_timesteps):
            timestep += 1
            action, log_prob, value = ppo.select_action(state)

            if continuous:
                action = np.clip(action, env.action_space.low, env.action_space.high)
                # Keep the stored old log-prob consistent with the action sent to the env.
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=ppo.device).unsqueeze(0)
                    action_tensor = torch.tensor(action, dtype=torch.float32, device=ppo.device).unsqueeze(0)
                    log_prob, _, _ = ppo.policy_old.evaluate(state_tensor, action_tensor)
                    log_prob = log_prob.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            memory.states.append(torch.tensor(state, dtype=torch.float32))
            if continuous:
                memory.actions.append(torch.tensor(action, dtype=torch.float32))
            else:
                memory.actions.append(torch.tensor([action], dtype=torch.int64))
            memory.logprobs.append(torch.tensor([log_prob], dtype=torch.float32))
            memory.rewards.append(reward)
            memory.values.append(value)
            memory.is_terminals.append(done)

            state = next_state

            if timestep >= config['update_timestep']:
                ppo.update(memory, current_episode=episode, max_episodes=config['max_episodes'])
                memory.clear_memory()
                timestep = 0

            elif done:
                if len(memory.states) > 0:
                    ppo.update(memory, current_episode=episode, max_episodes=config['max_episodes'])
                    memory.clear_memory()
                    timestep = 0

            if done:
                break

        rewards.append(episode_reward)
        average_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        print(f"Episode {episode}, Reward: {episode_reward:.1f}, Avg100: {average_reward:.1f}")

        if len(rewards) >= 100 and average_reward >= config['solved_reward']:
            solved_name = 'CartPole' if env_name == 'CartPole-v1' else 'LunarLander'
            print(f"{solved_name} solved in episode {episode}! Avg100={average_reward:.1f}")
            break

    env.close()
    return rewards


if __name__ == '__main__':
    print("Training on CartPole-v1")
    rewards_cartpole = train('CartPole-v1')

    print("Training on LunarLander-v3")
    rewards_lunar = train('LunarLander-v3')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_cartpole)
    plt.title('CartPole-v1 Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(rewards_lunar)
    plt.title('LunarLander-v3 Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('rewards.png')
    print("Plot saved as rewards.png")
    plt.show()
