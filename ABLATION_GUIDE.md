# PPO Ablation Guide

This project now includes a dedicated ablation runner for LunarLander-v3.

## Full Ablation Study

Run 5 seeds for the baseline, the GAE lambda sweep, and the entropy coefficient sweep:

```powershell
.\venv\Scripts\python.exe ablation.py --env LunarLander-v3 --seeds 0,1,2,3,4 --output-dir results\ablation_lunar
```

This creates:

- one JSON result per run
- one folder per experiment group
- mean/std shaded plots under `results\ablation_lunar\plots`
- `ablation_summary.json`

## Fast Smoke Test

Use this only to check that the pipeline works:

```powershell
.\venv\Scripts\python.exe ablation.py --env LunarLander-v3 --seeds 0 --max-episodes 6 --rollout-steps 64 --output-dir results\ablation_smoke
```

## Sweep Values

GAE lambda:

- `0.95`
- `0.98`
- `1.00`

Entropy coefficient:

- `0.0`
- `0.01`
- `0.05`

## Metrics Plotted

Each run and each seed-averaged group can produce separate plots for:

- reward / Avg100
- value loss
- explained variance
- approximate KL

