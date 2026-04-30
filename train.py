import gymnasium as gym
import torch
import numpy as np
from ppo import PPO, Memory
import matplotlib.pyplot as plt

def train(env_name, max_episodes=1000, max_timesteps=200, update_timestep=2000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True

    ppo = PPO(state_dim, action_dim, continuous)
    memory = Memory()

    timestep = 0
    rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            timestep += 1

            action, log_prob, _ = ppo.select_action(state)
            if continuous:
                action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            memory.states.append(torch.FloatTensor(state))
            if continuous:
                memory.actions.append(torch.FloatTensor(action))
            else:
                memory.actions.append(torch.FloatTensor([action]))
            memory.logprobs.append(torch.FloatTensor([log_prob]))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 195 and env_name == 'CartPole-v1':
            print("CartPole solved!")
            break
        elif len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200 and env_name == 'LunarLander-v2':
            print("LunarLander solved!")
            break

    env.close()
    return rewards

if __name__ == '__main__':
    # Train on CartPole
    print("Training on CartPole-v1")
    rewards_cartpole = train('CartPole-v1', max_episodes=500)

    # Train on LunarLander
    print("Training on LunarLander-v2")
    rewards_lunar = train('LunarLander-v2', max_episodes=1000)

    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_cartpole)
    plt.title('CartPole-v1 Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(rewards_lunar)
    plt.title('LunarLander-v2 Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('rewards.png')
    plt.show()