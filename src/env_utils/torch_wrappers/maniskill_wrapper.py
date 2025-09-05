from gymnasium import Wrapper
import jax
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.array(x)


class ManiSkillWrapper(Wrapper):
    """
    A wrapper for ManiSkill environments to ensure compatibility with the expected API.
    This wrapper is used to handle the ManiSkill environments in a way that is consistent
    with the other environments in the codebase.
    """

    def __init__(self, env, max_episode_steps: int, partial_reset):
        super().__init__(env)
        self.metadata = env.metadata
        self.asymmetric_obs = False
        self.max_episode_steps = max_episode_steps
        self.partial_reset = partial_reset

        self.returns = np.zeros(env.num_envs, dtype=np.float32)
        self.episode_len = np.zeros(env.num_envs, dtype=np.float32)
        self.success = np.zeros(env.num_envs, dtype=np.float32)

    @property
    def action_space(self):
        """
        Returns the action space of the environment.
        """
        return self.env.action_space

    @property
    def observation_space(self):
        """
        Returns the observation space of the environment.
        """
        return self.env.observation_space

    @property
    def single_observation_space(self):
        """
        Returns the observation space of a single environment.
        """
        return self.env.single_observation_space

    @property
    def single_action_space(self):
        """
        Returns the action space of a single environment.
        """
        return self.env.single_action_space

    @property
    def unwrapped(self):
        """
        Returns the underlying environment.
        """
        return self.env

    @property
    def num_actions(self):
        """
        Returns the number of actions in the action space.
        """
        return self.action_space.shape[1]

    @property
    def num_obs(self):
        """
        Returns the number of observations in the observation space.
        """
        return self.observation_space.shape[1]

    def reset(self, seed=None, options=dict()):
        """
        Resets the environment and returns the initial observation.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return jax.tree.map(to_numpy, obs), jax.tree.map(to_numpy, info)

    def step(self, action):
        """
        Takes a step in the environment with the given action.
        Returns the next observation, reward, done, and info.
        """
        action = torch.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = jax.tree.map(to_numpy, obs)
        reward = to_numpy(reward)
        terminated = to_numpy(terminated)
        truncated = to_numpy(truncated)
        info = jax.tree.map(to_numpy, info)

        if "final_info" in info:
            self.returns = (
                info["final_info"]["episode"]["return"]
                * info["_final_info"].astype(np.float32)
                + (1.0 - info["_final_info"].astype(np.float32)) * self.returns
            )
            self.episode_len = (
                info["final_info"]["episode"]["episode_len"]
                * info["_final_info"].astype(np.float32)
                + (1.0 - info["_final_info"].astype(np.float32)) * self.episode_len
            )
            self.success = (
                info["final_info"]["episode"]["success_once"]
                * info["_final_info"].astype(np.float32)
                + (1.0 - info["_final_info"].astype(np.float32)) * self.success
            )
        info["log_info"] = {
            "return": self.returns,
            "episode_len": self.episode_len,
            "success": self.success,
        }
        if self.partial_reset:
            # maniskill continues bootstrap on terminated, which playground does on truncated.
            # This unifies the interfaces in a very hacky way
            done = np.zeros_like(terminated, dtype=bool)
            truncated = np.logical_or(terminated, truncated)
        else:
            done = np.logical_or(terminated, truncated)
            truncated = np.zeros_like(done, dtype=bool)
        return obs, reward, done, truncated, info
