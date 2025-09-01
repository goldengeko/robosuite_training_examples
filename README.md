# ROBOSUITE_TRAINING_EXAMPLES

This repository contains example scripts for training reinforcement learning agents in [robosuite](https://robosuite.ai/docs/overview.html) environments using PyTorch.

## Contents

- `humanoid.py`: TD3 training loop for a composite humanoid robot in robosuite.
- `kinova_door.py`: TD3 training loop for Kinova3 robot on the Door environment.
- `networks.py`: Implementation of TD3, PPO, and supporting neural network classes.

Install dependencies:
```bash
pip install robosuite robosuite_models torch numpy
```

## Usage

Train TD3 on a humanoid robot:
```bash
python humanoid.py
```

Train TD3 on Kinova3 Door environment:
```bash
python kinova_door.py
```

## Notes

- Controller configs and robot names must match those available in your robosuite installation.
- TensorBoard logging is included but commented out; uncomment to enable.
- Modify episode count, environment, or agent hyperparameters as needed.

