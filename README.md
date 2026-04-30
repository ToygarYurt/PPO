# PPO Implementation

This project implements the Proximal Policy Optimization (PPO) algorithm from scratch using PyTorch. The implementation is tested on two benchmark environments from Gymnasium: CartPole-v1 (discrete action space) and LunarLander-v2 (continuous action space).

## Features

- PPO algorithm with actor-critic architecture
- Support for both discrete and continuous action spaces
- Training on multiple environments
- Reward plotting

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Box2D (for LunarLander)

## Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install torch gymnasium numpy matplotlib
   pip install "gymnasium[box2d]"
   ```

## Usage

Run the training script:
```
python train.py
```

This will train PPO on CartPole-v1 and then on LunarLander-v2, and generate a `rewards.png` plot.

## Files

- `ppo.py`: PPO algorithm implementation
- `train.py`: Training script
- `.gitignore`: Git ignore file
- `rewards.png`: Generated reward plot (after running)

## Algorithm Details

The PPO implementation includes:
- Actor network for policy
- Critic network for value estimation
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE) for advantage calculation
- Mini-batch updates

## Results

The algorithm is tested on:
- CartPole-v1: Solved when average reward over 100 episodes >= 195
- LunarLander-v2: Solved when average reward over 100 episodes >= 200

Training progress is printed to the console, and rewards are plotted at the end.