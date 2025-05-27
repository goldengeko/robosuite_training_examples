import gym
import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config, load_part_controller_config


class RobosuiteGymWrapper(gym.Env):
    def __init__(self, env_name="Door", robots="Kinova3", has_renderer=False):
        super().__init__()

        # Create the robosuite environment
        self.env = suite.make(
            env_name,
            robots=robots,
            controller_configs=[load_composite_controller_config(controller="HYBRID_MOBILE_BASE"),load_part_controller_config(default_controller="JOINT_POSITION")], 
            has_renderer=has_renderer,
            use_camera_obs=False,
            horizon=300,
            render_camera="frontview",
            reward_shaping=True,
            control_freq=20
        )

        # Define action and observation space
        self.action_space = gym.spaces.Box(
            low=self.env.action_spec[0], high=self.env.action_spec[1], dtype=np.float32
        )

        obs = self.env.reset()
        obs_space = np.concatenate([obs[k] for k in obs if isinstance(obs[k], np.ndarray)])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_space.shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs = np.concatenate([obs[k] for k in obs if isinstance(obs[k], np.ndarray)])
        return obs 


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([obs[k] for k in obs if isinstance(obs[k], np.ndarray)])
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()