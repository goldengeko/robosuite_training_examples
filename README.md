# ROBOSUITE_TRAINING_EXAMPLES

This repository contains example scripts for training reinforcement learning agents in [robosuite](https://robosuite.ai/docs/overview.html) environments using PyTorch.

## Contents

- `humanoid.py`: PPO training loop for a composite humanoid robot in robosuite using a whole-body IK controller.
- `kinova_door.py`: TD3 training loop for a custom Kinova3 robot on the Door environment.
- `alert_spot.py`: PPO training loop for a custom legged robot ("AlertSpot") combining Spot base and Kinova3 arm.
- `networks.py`: Implementation of TD3, PPO, and supporting neural network classes (actor, critic, replay buffer).
- `humanoid.json`: Example composite controller configuration for a humanoid robot using whole-body IK.

## Installation

Install dependencies (Python 3.10+ recommended):

```bash
pip install robosuite robosuite_models torch numpy mujoco
```

## Usage

### Train PPO on a humanoid robot:

```bash
python humanoid.py
```

- Uses the `humanoid.json` controller config.
- Robot: `"GR1FixedLowerBody"` (make sure this robot is available in your robosuite installation).
- Environment: `"Lift"`
- PPO agent, saving best actor/critic weights to `tmp/humanoid_ppo/`.

### Train TD3 on Kinova3 Door environment:

```bash
python kinova_door.py
```

- Uses a custom robot class `KinovaCustom` (inherits from Kinova3, with a fixed base and custom gripper).
- Environment: `"Door"`
- TD3 agent, TensorBoard logging enabled, saving best weights to `tmp/kinova_td3/`.

### Train PPO on a custom legged robot ("AlertSpot"):

```bash
python alert_spot.py
```

- Defines a custom robot class `AlertSpot` combining Spot base and Kinova3 arm.
- Environment: `"Lift"`
- PPO agent, saving best weights to `tmp/td3/`.

## Controller Configuration

- Composite controllers (e.g., `WHOLE_BODY_IK`) require a config JSON (see `humanoid.json`).
- Important keys:
  - `"ref_name"`: List of end-effector site names.
  - `"actuation_part_names"`: Body parts controlled by IK.
  - `"nullspace_joint_weights"`: Required for nullspace control (see error notes below).
  - `"ik_posture_weights"`: Optional, for posture cost.
  - `"body_parts"`: Per-part controller configs (arms, torso, head, grippers).

## Notes

- Controller configs and robot names must match those available in your robosuite installation.
- TensorBoard logging is included in `kinova_door.py` (uncomment in other scripts to enable).
- Modify episode count, environment, or agent hyperparameters as needed.
- If you encounter a `KeyError: 'nullspace_joint_weights'`, ensure your controller config includes the `"nullspace_joint_weights"` key in `"composite_controller_specific_configs"` (see `humanoid.json` for example).
- For custom robots, see the use of `register_robot_class` in `kinova_door.py` and `alert_spot.py`.

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

