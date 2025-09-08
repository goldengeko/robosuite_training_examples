# ROBOSUITE_TRAINING_EXAMPLES

This repository contains example scripts for training reinforcement learning agents in [robosuite](https://robosuite.ai/docs/overview.html) environments using PyTorch.

## Contents

- `humanoid.py`: PPO training loop for a composite humanoid robot in robosuite using a whole-body IK controller.
- `kinova_pick.py`: TD3 training loop for a custom Kinova3 robot on the Pick environment.
- `alert_spot.py`: PPO training loop for a custom legged robot ("AlertSpot") combining Spot base and Kinova3 arm.
- `networks.py`: Implementation of TD3, PPO, and supporting neural network classes (actor, critic, replay buffer).
- `humanoid.json`: Example composite controller configuration for a humanoid robot using whole-body IK.

## Installation

Install dependencies (Python 3.10+ recommended):

```bash
pip install robosuite robosuite_models torch numpy mujoco
```

## File Overview

| File             | Description                                                                                   |
|------------------|----------------------------------------------------------------------------------------------|
| `humanoid.py`    | PPO training loop for humanoid robot with whole-body IK controller.                          |
| `kinova_pick.py` | TD3 training loop for Kinova3 robot on Pick environment, with custom robot registration.      |
| `alert_spot.py`  | PPO training loop for custom legged robot combining Spot base and Kinova3 arm.               |
| `networks.py`    | Neural network implementations for TD3 and PPO agents, plus replay buffer.                   |
| `humanoid.json`  | Example composite controller config for humanoid robot (whole-body IK).                      |


# Add Kinova 6DOF robot to robosuite

- Copy the [kinova3_6dof](/kinova3_6dof/) folder to ```~/.local/lib/python3.10/site-packages/robosuite/models/assets/robots/```
- Copy the [controller configs](default_kinova3_6dof.json) to ``` ~/.local/lib/python3.10/site-packages/robosuite/controllers/config/robots/```

- Copy [Robot class script](kinova3_6dof_robot.py) to ```~/.local/lib/python3.10/site-packages/robosuite/models/robots/manipulators/```

- Add the following import statement to `~/.local/lib/python3.10/site-packages/robosuite/models/robots/manipulators/__init__.py`:

```python
from .kinova3_6dof_robot import Kinova3_6DOF
```

- Add the new robot to robot class mapping in ```~/.local/lib/python3.10/site-packages/robosuite/robots/__init__.py```

```python
"Kinova3_6DOF": FixedBaseRobot,
```

