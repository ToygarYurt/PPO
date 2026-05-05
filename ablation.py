import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import env_configs, plot_training_metrics, safe_name, train


DEFAULT_SEEDS = [0, 1, 2]
GAE_VALUES = [1.00]
ENTROPY_VALUES = [0.005]

# Ablation study variations focus on controlling the bias-variance tradeoff
# through the GAE lambda parameter and the exploration pressure via entropy
# regularization. These sweeps help measure sensitivity of PPO training
# dynamics for LunarLander-v3.


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def pad_series(series_list):
    max_len = max(len(series) for series in series_list)
    padded = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for row, series in enumerate(series_list):
        padded[row, : len(series)] = np.asarray(series, dtype=np.float64)
    return padded


def mean_std(series_list):
    padded = pad_series(series_list)
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value)


def run_seed_group(env_name, label, overrides, seeds, output_dir, verbose=False):
    group_dir = output_dir / safe_name(label)
    group_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    # Execute multiple random seeds for a single experimental condition to
    # quantify variability and produce statistically meaningful metrics.

    for seed in seeds:
        run_name = f"{safe_name(label)}_seed{seed}"
        print(f"[{label}] seed={seed}")
        result = train(
            env_name,
            seed=seed,
            config_overrides=overrides,
            output_dir=group_dir,
            run_name=run_name,
            verbose=verbose,
        )
        result_path = group_dir / f"{run_name}.json"
        save_json(result, result_path)
        plot_training_metrics(result, output_dir=group_dir, prefix=run_name)
        runs.append(result)

    save_json({"label": label, "env_name": env_name, "overrides": overrides, "runs": runs}, group_dir / "group_results.json")
    return runs


def extract_series(result, metric):
    if metric == "reward":
        return result["moving_avg_100"]
    return result.get("metrics", {}).get(metric, [])


def plot_seed_mean(runs, metric, title, ylabel, save_path):
    series_list = [extract_series(run, metric) for run in runs if extract_series(run, metric)]
    if not series_list:
        return
    mean, std = mean_std(series_list)
    x = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, linewidth=2, label="mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.22, label="std")
    plt.xlabel("Episode" if metric == "reward" else "PPO update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sweep_comparison(groups, metric, title, ylabel, save_path):
    plt.figure(figsize=(11, 6))
    for label, runs in groups.items():
        series_list = [extract_series(run, metric) for run in runs if extract_series(run, metric)]
        if not series_list:
            continue
        mean, std = mean_std(series_list)
        x = np.arange(1, len(mean) + 1)
        plt.plot(x, mean, linewidth=2, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.14)

    plt.xlabel("Episode" if metric == "reward" else "PPO update")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def summarize_groups(groups):
    summary = {}
    for label, runs in groups.items():
        best_values = [run["metadata"]["best_average_reward"] for run in runs]
        final_values = [run["metadata"]["final_100_avg"] for run in runs]
        finite_best = [value for value in best_values if np.isfinite(value)]
        summary[label] = {
            "best_avg100_mean": float(np.mean(finite_best)) if finite_best else None,
            "best_avg100_std": float(np.std(finite_best)) if finite_best else None,
            "final_avg100_mean": float(np.mean(final_values)),
            "final_avg100_std": float(np.std(final_values)),
            "num_seeds": len(runs),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run PPO ablation studies on LunarLander-v3.")
    parser.add_argument("--env", default="LunarLander-v3", choices=list(env_configs.keys()))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional override for faster pilot runs.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Optional override for quick smoke tests.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_overrides = {}
    if args.max_episodes is not None:
        common_overrides["max_episodes"] = args.max_episodes
        common_overrides["min_episodes"] = min(env_configs[args.env].get("min_episodes", 100), args.max_episodes)
    if args.rollout_steps is not None:
        common_overrides["rollout_steps"] = args.rollout_steps

    groups = {}

    baseline_label = "baseline"
    groups[baseline_label] = run_seed_group(
        args.env,
        baseline_label,
        common_overrides,
        seeds,
        output_dir,
        verbose=args.verbose,
    )

    for value in GAE_VALUES:
        label = f"gae_lambda_{value:.2f}"
        overrides = {**common_overrides, "gae_lambda": value}
        groups[label] = run_seed_group(args.env, label, overrides, seeds, output_dir, verbose=args.verbose)

    for value in ENTROPY_VALUES:
        label = f"entropy_coef_{value:.2f}"
        overrides = {**common_overrides, "entropy_coef": value}
        groups[label] = run_seed_group(args.env, label, overrides, seeds, output_dir, verbose=args.verbose)

    # Plotting both the mean and variance across seeds emphasizes not just the
    # average effect of each parameter setting but also the stability of RL
    # performance under different random initializations.

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_info = {
        "reward": ("Reward Avg100", "Avg100 reward"),
        "value_loss": ("Value Loss", "value loss"),
        "explained_variance": ("Explained Variance", "explained variance"),
        "approx_kl": ("Approximate KL", "approx KL"),
    }

    for label, runs in groups.items():
        for metric, (title, ylabel) in metric_info.items():
            plot_seed_mean(
                runs,
                metric,
                f"{args.env} {label} {title}",
                ylabel,
                plot_dir / f"{safe_name(label)}_{metric}_mean_std.png",
            )

    gae_groups = {label: runs for label, runs in groups.items() if label.startswith("gae_lambda")}
    entropy_groups = {label: runs for label, runs in groups.items() if label.startswith("entropy_coef")}

    for metric, (title, ylabel) in metric_info.items():
        plot_sweep_comparison(
            gae_groups,
            metric,
            f"{args.env} GAE lambda sweep: {title}",
            ylabel,
            plot_dir / f"gae_lambda_sweep_{metric}.png",
        )
        plot_sweep_comparison(
            entropy_groups,
            metric,
            f"{args.env} entropy coefficient sweep: {title}",
            ylabel,
            plot_dir / f"entropy_coef_sweep_{metric}.png",
        )

    summary = summarize_groups(groups)
    save_json(summary, output_dir / "ablation_summary.json")
    print(f"Ablation complete. Results saved under {output_dir}")


if __name__ == "__main__":
    main()
