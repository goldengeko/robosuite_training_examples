from robosuite.robots import register_robot_class
from robosuite.models.robots import Kinova3
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from networks import PPO, ReplayBuffer
import numpy as np
import torch
import mujoco

@register_robot_class("LeggedRobot")
class AlertSpot(Kinova3):
    @property
    def default_base(self):
        return "Spot"
    
    def set_base_xpos(self, pos):
        return super().set_base_xpos([pos[1] - 0.75, pos[0], pos[2]])

    @property
    def default_arms(self):
        return {"right": "Kinova3"}
    @property
    def default_gripper(self):
        return {"right": "Robotiq140Gripper"}
    

# Create environment
env = suite.make(
    env_name="Lift",
    robots=["AlertSpot"],

    controller_configs=[load_composite_controller_config(
        robot="SpotArm"
    )],
    reward_shaping=True,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="frontview",
    use_camera_obs=False,
    control_freq=20
)

# env.reset()
# mujoco.viewer.launch(env.sim.model._model, env.sim.data._data)

env = GymWrapper(env)

obs, _ = env.reset()
if isinstance(obs, dict):
    state = np.concatenate([v.ravel() for v in obs.values()])
else:
    state = obs

state_dim = state.shape[0]
action_dim = env.action_spec[0].shape[0]
max_action = 1.0

replay_buffer = ReplayBuffer()
agent = PPO(state_dim, action_dim, max_action)

#--------------Training Loop--------------
episodes = 100
episode_length = 500

best_reward = -float('inf')

for ep in range(episodes):
    obs, _ = env.reset()
    state = np.concatenate([v.ravel() for v in obs.values()]) if isinstance(obs, dict) else obs

    ep_reward = 0

    for t in range(episode_length):
        action, _ = agent.select_action(state)
        action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        next_state = np.concatenate([v.ravel() for v in next_obs.values()]) if isinstance(next_obs, dict) else next_obs
        replay_buffer.add((state, action, reward, next_state, float(done)))

        if len(replay_buffer.storage) > 5000:
            agent.train(replay_buffer, batch_size=100)

        state = next_state

        if done:
            break

    print(f"Episode {ep}: Total Reward = {ep_reward}")

    if ep_reward > best_reward:
        best_reward = ep_reward
        torch.save(agent.actor.state_dict(), "tmp/td3/best_actor.pth")
        torch.save(agent.critic.state_dict(), "tmp/td3/best_critic.pth")
