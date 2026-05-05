# LunarLander-v3 Diagnosis

Latest run summary:

- Seed: `18`
- Episodes: `1779`
- Best Avg100: `116.18` at episode `1279`
- Final Avg100: `90.35`
- Stochastic evaluation mean: `112.29`
- Deterministic evaluation mean: `64.40`

## Interpretation

The run is not a catastrophic collapse. The policy improves from about `-162` Avg100 at the beginning to `116` Avg100, then drifts down by about `26` points. This is normal variance for PPO on LunarLander, but the diagnostic metrics show the policy updates are too conservative:

- Approx KL stays around `0.002`, far below the KL limit.
- Clip fraction stays around `0.02`, so PPO clipping almost never activates.
- Value loss decreases strongly.
- Explained variance rises to about `0.80`, meaning the critic is learning the value function.

So the critic is learning, but the actor is not moving aggressively enough to reach the stronger `160-200` Avg100 range.

## Config Adjustment

Only LunarLander-v3 was changed:

- Learning rate: `7e-5 -> 1.5e-4`
- Target KL: `0.008 -> 0.015`
- Target reward: `220 -> 160`
- Patience: `500 -> 300`

Acrobot-v1 was not changed.

## Recommended Next Run

```powershell
.\venv\Scripts\python.exe train.py --env LunarLander-v3 --seed 18
```

If the result is still below `130` best Avg100, try:

```powershell
.\venv\Scripts\python.exe train.py --env LunarLander-v3 --seed 7
```

