import argparse
import copy
import json
import time
from pathlib import Path
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ppo import Memory, PPO

env_configs = {
    "CartPole-v1": {
        "max_episodes": 600,
        "rollout_steps": 2048,
        "lr": 1e-3,
        "K_epochs": 4,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "eps_clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "target_reward": 475.0,
        "patience": 150,
        "min_episodes": 100,
        "target_kl": 0.03,
        "eval_deterministic": True,
    },
    "LunarLander-v3": {
        "max_episodes": 3000,
        "rollout_steps": 4096,
        "lr": 7e-5,
        "K_epochs": 4,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "eps_clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.005,
        "target_reward": 220.0,
        "patience": 500,
        "min_episodes": 500,
        "target_kl": 0.008,
        "eval_deterministic": False,
        "normalize_observations": True,
        "degradation_stop_reward": 40.0,
        "degradation_margin": 70.0,
        "degradation_patience": 80,
    },
    "Acrobot-v1": {
        "max_episodes": 1200,
        "rollout_steps": 2048,
        "lr": 3e-4,
        "K_epochs": 4,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "eps_clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "target_reward": -100.0,
        "patience": 250,
        "min_episodes": 200,
        "target_kl": 0.02,
        "eval_deterministic": True,
        "normalize_observations": True,
    },
}

class RunningMeanStd:
    """Tracks observation statistics for online normalization."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x, update=False, clip=10.0):
        if update:
            self.update(np.asarray(x, dtype=np.float64))
        normalized = (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self):
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def moving_average(values, window=100):
    if len(values) == 0:
        return []
    return [float(np.mean(values[max(0, i - window + 1): i + 1])) for i in range(len(values))]


def build_config(env_name, overrides=None):
    config = copy.deepcopy(env_configs[env_name])
    if overrides:
        config.update(overrides)
    return config


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

def make_agent(env, config):
    state_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = int(env.action_space.n)
        continuous = False
    else:
        action_dim = int(np.prod(env.action_space.shape))
        continuous = True

    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        lr=config["lr"],
        gamma=config["gamma"],
        eps_clip=config["eps_clip"],
        K_epochs=config["K_epochs"],
        batch_size=config["batch_size"],
        gae_lambda=config["gae_lambda"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
        target_kl=config.get("target_kl"),
    )
    return ppo, continuous, state_dim, action_dim

def store_transition(memory, state, action, log_prob, reward, value, done, continuous):
    memory.states.append(torch.tensor(state, dtype=torch.float32))
    if continuous:
        memory.actions.append(torch.tensor(action, dtype=torch.float32))
    else:
        memory.actions.append(torch.tensor([action], dtype=torch.int64))
    memory.logprobs.append(torch.tensor([log_prob], dtype=torch.float32))
    memory.rewards.append(float(reward))
    memory.values.append(float(value))
    memory.is_terminals.append(bool(done))

def evaluate(env_name, checkpoint_path, episodes=10, seed=123, deterministic=True):
    env = gym.make(env_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo, continuous, _, _ = make_agent(env, checkpoint["config"])
    ppo.policy.load_state_dict(checkpoint["policy_state_dict"])
    ppo.policy_old.load_state_dict(checkpoint["policy_old_state_dict"])
        
    obs_rms = None
    if checkpoint["config"].get("normalize_observations") and checkpoint.get("obs_rms") is not None:
        obs_rms = RunningMeanStd(env.observation_space.shape)
        obs_rms.load_state_dict(checkpoint["obs_rms"])


    rewards = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        while not done:
            model_state = obs_rms.normalize(state, update=False) if obs_rms is not None else state
            action, _, _ = ppo.select_action(model_state, deterministic=deterministic)
            if continuous:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(float(episode_reward))

    env.close()
    return {
        "episodes": episodes,
        "mode": "deterministic" if deterministic else "stochastic",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "rewards": rewards,
    }

def train(env_name, seed=42, config_overrides=None, output_dir=".", run_name=None, verbose=True):
    config = build_config(env_name, config_overrides)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or env_name.replace("-", "_")
    env = gym.make(env_name)
    env.action_space.seed(seed)
    ppo, continuous, state_dim, action_dim = make_agent(env, config)
    obs_rms = RunningMeanStd(env.observation_space.shape) if config.get("normalize_observations") else None
    memory = Memory()

    rewards = []
    metrics = {
        "episode": [],
        "update": [],
        "value_loss": [],
        "explained_variance": [],
        "approx_kl": [],
        "policy_loss": [],
        "entropy": [],
        "clip_fraction": [],
    }
    best_average_reward = -float("inf")
    best_episode = 0
    degradation_start_episode = None

    checkpoint_path = output_dir / f"best_{safe_name(run_name)}.pth"
    start_time = time.time()
    update_count = 0

    for episode in range(1, config["max_episodes"] + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            model_state = obs_rms.normalize(state, update=True) if obs_rms is not None else state
            action, log_prob, value = ppo.select_action(model_state)
            if continuous:
                action = np.clip(action, env.action_space.low, env.action_space.high)
                with torch.no_grad():
                    state_tensor = torch.tensor(model_state, dtype=torch.float32, device=ppo.device).unsqueeze(0)
                    action_tensor = torch.tensor(action, dtype=torch.float32, device=ppo.device).unsqueeze(0)
                    log_prob, _, _ = ppo.policy_old.evaluate(state_tensor, action_tensor)
                    log_prob = log_prob.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            store_transition(memory, model_state, action, log_prob, reward, value, done, continuous)
            state = next_state

            if len(memory.rewards) >= config["rollout_steps"]:
                if done:
                    bootstrap_value = 0.0
                else:
                    bootstrap_state = obs_rms.normalize(next_state, update=False) if obs_rms is not None else next_state
                    bootstrap_value = ppo.get_value(bootstrap_state)
                update_metrics = ppo.update(
                    memory,
                    current_episode=episode,
                    max_episodes=config["max_episodes"],
                    next_value=bootstrap_value,
                )
                update_count += 1
                metrics["episode"].append(episode)
                metrics["update"].append(update_count)
                for key in ("value_loss", "explained_variance", "approx_kl", "policy_loss", "entropy", "clip_fraction"):
                    metrics[key].append(float(update_metrics[key]))
                memory.clear_memory()

        rewards.append(float(episode_reward))
        avg100 = float(np.mean(rewards[-100:]))

        if len(rewards) >= 100 and avg100 > best_average_reward:
            best_average_reward = avg100
            best_episode = episode
            degradation_start_episode = None

            torch.save(
                {
                    "episode": episode,
                    "average_reward": best_average_reward,
                    "policy_state_dict": ppo.policy.state_dict(),
                    "policy_old_state_dict": ppo.policy_old.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "config": config,
                    "env_name": env_name,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "continuous": continuous,
                    "seed": seed,
                    "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,

                },
                checkpoint_path,
            )
            if verbose:
                print(f"New best {env_name}: episode={episode}, Avg100={best_average_reward:.1f}")

        if verbose:
            print(f"{env_name} | Episode {episode}, Reward: {episode_reward:.1f}, Avg100: {avg100:.1f}")

        if len(rewards) >= 100 and avg100 >= config["target_reward"]:
            if verbose:
                print(f"{env_name} reached target at episode {episode}: Avg100={avg100:.1f}")
            break

        if len(rewards) >= config.get("min_episodes", 100):
            if best_average_reward >= config["target_reward"]:
                if verbose:
                    print(
                        f"{env_name} early stopped at episode {episode}: "
                        f"best Avg100={best_average_reward:.1f} at episode {best_episode}"
                    )
                break
            degradation_stop_reward = config.get("degradation_stop_reward")
            degradation_margin = config.get("degradation_margin", 70.0)
            degradation_patience = config.get("degradation_patience", 100)
            if degradation_stop_reward is not None and best_average_reward >= degradation_stop_reward:
                if avg100 <= best_average_reward - degradation_margin:
                    if degradation_start_episode is None:
                        degradation_start_episode = episode
                    elif episode - degradation_start_episode >= degradation_patience:
                        if verbose:
                            print(
                                f"{env_name} stopped after performance degradation at episode {episode}: "
                                f"current Avg100={avg100:.1f}, best Avg100={best_average_reward:.1f} "
                                f"at episode {best_episode}"
                            )
                        break
                else:
                    degradation_start_episode = None
            if episode - best_episode >= config.get("patience", config["max_episodes"]):
                if verbose:
                    print(
                        f"{env_name} stopped by patience at episode {episode}: "
                        f"best Avg100={best_average_reward:.1f} at episode {best_episode}"
                    )
                break

    if len(memory.rewards) > 0:
        update_metrics = ppo.update(memory, current_episode=len(rewards), max_episodes=config["max_episodes"], next_value=0.0)
        update_count += 1
        metrics["episode"].append(len(rewards))
        metrics["update"].append(update_count)
        for key in ("value_loss", "explained_variance", "approx_kl", "policy_loss", "entropy", "clip_fraction"):
            metrics[key].append(float(update_metrics[key]))
        memory.clear_memory()

    env.close()
    elapsed = time.time() - start_time

    result = {
        "metadata": {
            "environment": env_name,
            "total_episodes": len(rewards),
            "best_average_reward": float(best_average_reward),
            "final_100_avg": float(np.mean(rewards[-100:])),
            "training_time_seconds": float(elapsed),
            "action_type": "Continuous" if continuous else "Discrete",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "checkpoint": str(checkpoint_path),
            "seed": seed,
            "run_name": run_name,
        },
        "rewards": rewards,
        "moving_avg_100": moving_average(rewards, 100),
        "metrics": metrics,
        "config": config,
    }

    if best_average_reward > -float("inf"):
        primary_deterministic = config.get("eval_deterministic", True)
        result["evaluation"] = evaluate(
            env_name,
            checkpoint_path,
            episodes=20,
            seed=seed + 10_000,
            deterministic=primary_deterministic,
        )
        result["evaluation_comparison"] = {
            "deterministic": evaluate(
                env_name,
                checkpoint_path,
                episodes=10,
                seed=seed + 20_000,
                deterministic=True,
            ),
            "stochastic": evaluate(
                env_name,
                checkpoint_path,
                episodes=10,
                seed=seed + 30_000,
                deterministic=False,
            ),
        }

    return result

def plot_results(results, save_path="ppo_training_results.png"):
    plt.figure(figsize=(12, 5))
    for index, (env_name, data) in enumerate(results.items(), start=1):
        plt.subplot(1, len(results), index)
        plt.plot(data["rewards"], alpha=0.35, label="Episode reward")
        plt.plot(data["moving_avg_100"], linewidth=2, label="Avg100")
        best_avg = data.get("metadata", {}).get("best_average_reward")
        if best_avg is not None:
            best_episode = int(np.argmax(data["moving_avg_100"]))
            plt.scatter(best_episode, best_avg, color="tab:red", s=35, zorder=3, label="Best Avg100")
            plt.annotate(
                f"best={best_avg:.1f}",
                xy=(best_episode, best_avg),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
            )
        plt.title(env_name)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.25)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {save_path}")


def plot_training_metrics(result, output_dir=".", prefix=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    prefix = safe_name(prefix or metadata.get("run_name", metadata["environment"]))

    reward_path = output_dir / f"{prefix}_reward.png"
    plt.figure(figsize=(10, 5))
    plt.plot(result["rewards"], alpha=0.35, label="Episode reward")
    plt.plot(result["moving_avg_100"], linewidth=2, label="Avg100")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{metadata['environment']} reward")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_path, dpi=150)
    plt.close()

    metric_specs = {
        "value_loss": ("Value Loss", "value loss"),
        "explained_variance": ("Explained Variance", "explained variance"),
        "approx_kl": ("Approximate KL", "approx KL"),
    }
    metrics = result.get("metrics", {})
    episodes = metrics.get("episode", [])
    saved_paths = {"reward": str(reward_path)}

    for key, (title, ylabel) in metric_specs.items():
        values = metrics.get(key, [])
        if not episodes or not values:
            continue
        save_path = output_dir / f"{prefix}_{key}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, values, linewidth=1.8)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{metadata['environment']} {title}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        saved_paths[key] = str(save_path)

    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Train PPO on benchmark Gymnasium environments.")
    parser.add_argument("--env", choices=list(env_configs.keys()) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results", help="Directory for JSON, checkpoints, and plots.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_names = list(env_configs.keys()) if args.env == "all" else [args.env]
    results = {}

    for env_name in env_names:
        print(f"\nTraining PPO on {env_name}")
        run_name = f"{env_name.replace('-', '_')}_seed{args.seed}"
        results[env_name] = train(env_name, seed=args.seed, output_dir=output_dir, run_name=run_name)
        plot_training_metrics(results[env_name], output_dir=output_dir, prefix=run_name)

    results_path = output_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value)

    plot_results(results, save_path=output_dir / "ppo_training_results.png")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
