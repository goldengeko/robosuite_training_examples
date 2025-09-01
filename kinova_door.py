import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from networks import TD3, ReplayBuffer
import torch
from torch.utils.tensorboard import SummaryWriter

if not os.path.exists("tmp/td3"):
    os.makedirs("tmp/td3")

writer = SummaryWriter(log_dir="runs/td3_training")

env_name = "Door"
env = suite.make(
    env_name,
    robots=["Kinova3"],
    controller_configs=[load_composite_controller_config(
        controller="/home/user/.local/lib/python3.10/site-packages/robosuite/controllers/config/robots/default_kinova3.json"
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

#--------------Training Loop--------------
episodes = 100
episode_length = 500

best_reward = -float('inf')

for ep in range(episodes):
    obs, _ = env.reset()
    state = np.concatenate([v.ravel() for v in obs.values()]) if isinstance(obs, dict) else obs

    ep_reward = 0

    for t in range(episode_length):
        action = agent.select_action(state)
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

    writer.add_scalar("Reward/Episode", ep_reward, ep)

    for name, param in agent.actor.named_parameters():
        writer.add_histogram(f"Actor/{name}", param, ep)
    for name, param in agent.critic.named_parameters():
        writer.add_histogram(f"Critic/{name}", param, ep)

    print(f"Episode {ep}: Total Reward = {ep_reward}")

    if ep_reward > best_reward:
        best_reward = ep_reward
        torch.save(agent.actor.state_dict(), "tmp/td3/best_actor.pth")
        torch.save(agent.critic.state_dict(), "tmp/td3/best_critic.pth")


#------------Inference----------------

# Load the trained actor and critic networks
# agent.actor.load_state_dict(torch.load("tmp/td3/actor.pth"))
# # agent.critic.load_state_dict(torch.load("tmp/td3/critic.pth"))

# agent.actor.eval()
# # agent.critic.eval()

# obs, _ = env.reset()
# done = False

# while not done:
#     state = torch.tensor(obs, dtype=torch.float32).to(device="cpu")
#     action = agent.actor(state).cpu().detach().numpy().flatten()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

    
