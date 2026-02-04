from gymnasium import Wrapper, spaces
import jax
import jax.numpy as jnp
import numpy as np
import torch

def to_jax(x):
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    elif isinstance(x, jax.Array):
        return x
    elif isinstance(x, torch.Tensor):
        return jnp.array(x.cpu().numpy()) # not zero-copy though
        # return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
        # The error occurs because jax.dlpack.from_dlpack expects an object that implements the dlpack and dlpack_device methods (i.e., a DLPack-compatible tensor), but torch.utils.dlpack.to_dlpack(x.contiguous()) returns a DLPack capsule, not an object with those methods.
    elif isinstance(x, dict) or isinstance(x, list):
        return jax.tree.map(to_jax, x)
    else:
        return jnp.array(x)


def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, jax.Array):
        return torch.from_numpy(np.array(x))
        # return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x)) # no function called to_dlpack in jax.dlpack
    else:
        raise ValueError(f"Cannot convert type {type(x)} to torch.Tensor")


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

        self.returns = jnp.zeros(env.num_envs, dtype=np.float32)
        self.episode_len = jnp.zeros(env.num_envs, dtype=np.float32)
        self.success = jnp.zeros(env.num_envs, dtype=np.float32)
        
        # Cache the overridden observation spaces to force them to be used
        # from gymnasium import spaces
        # try:
        #     num_envs = env.num_envs
        # except AttributeError:
        #     num_envs = 1

    @property
    def action_space(self):
        """
        Returns the action space of the environment.
        """
        return self.env.action_space

    @property
    def observation_space(self):
        # return self.env.observation_space
        """
        Returns the observation space of the environment.
        After filtering, observations are 25-dimensional.
        """
        # Return a 25-dim Box space (num_envs, 25)
        try:
            num_envs = self.env.num_envs
        except AttributeError:
            num_envs = 1
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_envs, 25),
            dtype=np.float32
        )

    @property
    def single_observation_space(self):
        # return self.env.single_observation_space
        """
        Returns the observation space of a single environment.
        After filtering, observations are 25-dimensional.
        """
        # Return a 25-dim Box space
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

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
        return to_jax(obs), to_jax(info)

    def step(self, action):
        """
        Takes a step in the environment with the given action.
        Returns the next observation, reward, done, and info.
        """
        # Scale action from [-1, 1] to [action_low, action_high]
        action_low = np.array(self.action_space.low)
        action_high = np.array(self.action_space.high)
        
        # Check if action space is not [-1, 1]
        if not (np.allclose(action_low, -1) and np.allclose(action_high, 1)):
            action = action_low + (action + 1.0) / 2.0 * (action_high - action_low)
        
        action = to_torch(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = to_jax(obs)
        reward = to_jax(reward)
        terminated = to_jax(terminated)
        truncated = to_jax(truncated)
        info = to_jax(info)

        if "final_info" in info:
            self.returns = (
                info["final_info"]["episode"]["return"] * info["_final_info"]
                + (1.0 - info["_final_info"]) * self.returns
            )
            self.episode_len = (
                info["final_info"]["episode"]["episode_len"] * info["_final_info"]
                + (1.0 - info["_final_info"]) * self.episode_len
            )
            self.success = (
                info["final_info"]["episode"]["success_once"] * info["_final_info"]
                + (1.0 - info["_final_info"]) * self.success
            )
        info["log_info"] = {
            "return": self.returns,
            "episode_len": self.episode_len,
            "success": self.success,
        }
        if self.partial_reset:
            # maniskill continues bootstrap on terminated, which playground does on truncated.
            # This unifies the interfaces in a very hacky way
            done = jnp.zeros_like(terminated, dtype=bool)
            truncated = jnp.logical_or(terminated, truncated)
        else:
            done = jnp.logical_or(terminated, truncated)
            truncated = jnp.zeros_like(done, dtype=bool)
        return obs, reward, done, truncated, info
