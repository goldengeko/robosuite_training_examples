import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from networks import TD3, ReplayBuffer

if not os.path.exists("tmp/td3"):
    os.makedirs("tmp/td3")
    
env_name = "Door"
env = suite.make(
    env_name,
    robots=["Kinova3"],
    controller_configs=[load_composite_controller_config(
        controller="/home/guts/robosuite/lib/python3.10/site-packages/robosuite/controllers/config/robots/default_kinova3.json"
    )],
    reward_shaping=True,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="frontview",
    use_camera_obs=False,
    control_freq=20
)
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
agent = TD3(state_dim, action_dim, max_action)

episodes = 1000
episode_length = 500

for ep in range(episodes):
    obs, _ = env.reset()
    if isinstance(obs, dict):
        state = np.concatenate([v.ravel() for v in obs.values()])
    else:
        state = obs

    ep_reward = 0

    for t in range(episode_length):
        action = agent.select_action(state)
        action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        if isinstance(next_obs, dict):
            next_state = np.concatenate([v.ravel() for v in next_obs.values()])
        else:
            next_state = next_obs

        replay_buffer.add((state, action, reward, next_state, float(done)))

        if len(replay_buffer.storage) > 5000:
            agent.train(replay_buffer, batch_size=100)

        state = next_state
        env.render()

        if done:
            break

    print(f"Episode {ep}: Total Reward = {ep_reward}")
