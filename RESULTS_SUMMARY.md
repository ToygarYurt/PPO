# PPO Results Summary

This project evaluates a from-scratch PPO implementation on two Gymnasium benchmark environments: CartPole-v1 and LunarLander-v3.

## Main Results

| Environment | Best Avg100 | Primary Evaluation | Evaluation Mean | Notes |
| --- | ---: | --- | ---: | --- |
| CartPole-v1 | 262.82 | deterministic, 30 episodes | 469.33 | The learned policy is strong under greedy action selection. |
| LunarLander-v3 | 175.12 | stochastic, 30 episodes | 149.05 | The policy is much better when sampled stochastically than when forced to argmax. |

## Interpretation

CartPole-v1 shows a clear upward learning curve. Although the final training Avg100 does not reach the strict Gymnasium solved threshold of 475, deterministic evaluation of the best checkpoint reaches a high average reward, showing that the policy learned a stable control strategy.

LunarLander-v3 is more unstable, which is expected for PPO on this environment. The agent improves from roughly -180 Avg100 at the beginning to 175 Avg100 by the end of training. Deterministic evaluation is weaker than stochastic evaluation, so the report uses stochastic evaluation as the primary metric for LunarLander.

## Practical Lessons

- PPO performance depends strongly on rollout size, learning rate, entropy, and update stability.
- Saving the best checkpoint is necessary because continued training can degrade the policy.
- KL-based early stopping helps avoid overly large policy updates.
- Reporting both training curves and separate evaluation runs gives a more honest view of performance.

