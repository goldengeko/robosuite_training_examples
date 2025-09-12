import os
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from networks import TD3, ReplayBuffer
import torch
from robosuite.environments.manipulation.lift import Lift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("tmp/kinova_td3"):
    os.makedirs("tmp/kinova_td3")


class CustomLift(Lift):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cube_body = self.sim.model.body_name2id("cube_main")
        self.cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        self.table_height = self.model.mujoco_arena.table_offset[2]
        self.gripper = self.robots[0].gripper

    def reward(self, action=None):

        reward = 0.0

        
        """
        Type 1: sparse reward: This type of reward is generally harder to learn from, but is more
        representative of real-world tasks where a binary success signal is
        often the only available feedback.

        if self._check_success():
            reward = 1.0
        
        """

        """
        Type 2: dense reward: This type of reward provides more frequent feedback to the agent,
        which can help speed up learning.


        elif self.reward_shaping:

            # reaching reward
            TODO: You can modify the reaching reward as needed

            # grasping reward
            TODO: You can modify the grasping reward as needed

        Scale reward if requested: scaling is often useful for algorithms 
        that are sensitive to the magnitude of the reward signal.


        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
 
            
        """

        return reward


    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # return TODO: define the success condition 


env = CustomLift(
    robots="Kinova3_6DOF",
    controller_configs=[load_composite_controller_config(
        robot="Kinova3_6DOF",
    )],
    # reward_shaping= TODO: Set to True or False,
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
# episodes = TODO: Set the number of episodes
# episode_length = TODO: Set the maximum length of each episode

best_reward = -float('inf')

for ep in range(episodes):
    obs, _ = env.reset()
    state = np.concatenate([v.ravel() for v in obs.values()]) if isinstance(obs, dict) else obs

    ep_reward = 0
    losses = []

    for t in range(episode_length):
        action = agent.select_action(state)
        action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)
        print(f"Action space: {action}") 

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        next_state = np.concatenate([v.ravel() for v in next_obs.values()]) if isinstance(next_obs, dict) else next_obs
        replay_buffer.add((state, action, reward, next_state, float(done)))

        if len(replay_buffer.storage) > 5000:
            loss = agent.train(replay_buffer, batch_size=100)
            if loss is not None:
                losses.append(loss)

        state = next_state

        if done:
            break

    print(f"Episode {ep}: Total Reward = {ep_reward}")

    if ep_reward > best_reward:
        best_reward = ep_reward
        # TODO: Save the best model (Should the actor be saved or the critic or both?)

#------------Inference----------------

# Load the trained actor and critic networks
agent.actor.load_state_dict(torch.load("tmp/kinova_td3/best_actor.pth", map_location=device))
agent.actor.eval()

# Run inference
done = False
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    action = agent.actor(state_tensor.unsqueeze(0)).cpu().detach().numpy().flatten()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = np.concatenate([v.ravel() for v in obs.values()]) if isinstance(obs, dict) else obs
