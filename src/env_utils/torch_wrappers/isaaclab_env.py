from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
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
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
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
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
    else:
        raise ValueError(f"Cannot convert type {type(x)} to torch.Tensor")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabEnv:
    """Wrapper for IsaacLab environments to be compatible with MuJoCo Playground"""

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[list] = None,
    ):
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.unwrapped.single_observation_space
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.unwrapped.single_observation_space[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]
        
        # Create gymnasium Box spaces for compatibility
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0 if action_bounds is None else action_bounds[0],
            high=1.0 if action_bounds is None else action_bounds[1],
            shape=(self.num_actions,),
            dtype=np.float32,
        )
        
        # Tracking variables for logging
        self.returns = jnp.zeros(num_envs, dtype=jnp.float32)
        self.episode_len = jnp.zeros(num_envs, dtype=jnp.float32)
        self.current_returns = jnp.zeros(num_envs, dtype=jnp.float32)
        self.current_episode_len = jnp.zeros(num_envs, dtype=jnp.float32)

        self.has_been_reset = False

    def reset(self, random_start_init: bool = True) -> jax.Array:
        if self.has_been_reset:
            raise RuntimeError(
                "Environment has already been reset once. "
                "Please create a new environment instance for a fresh start."
            )
        self.has_been_reset = True
        obs_dict, _ = self.envs.reset()
        # NOTE: decorrelate episode horizons like RSL‑RL
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        return to_jax(obs_dict["policy"]), None

    def reset_with_critic_obs(self) -> tuple[jax.Array, jax.Array]:
        obs_dict, _ = self.envs.reset()
        return to_jax(obs_dict["policy"]), to_jax(obs_dict["critic"])

    def step(
        self, actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
        # Convert actions from jax to torch
        actions = to_torch(actions)
        if self.action_bounds is not None:
            actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions)
        dones = (terminations).to(dtype=torch.long)
        obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"] if self.asymmetric_obs else None
        
        # Convert to jax for tracking
        rew_jax = to_jax(rew)
        dones_jax = to_jax(dones)
        truncations_jax = to_jax(truncations)
        
        # Update current episode tracking
        self.current_returns = self.current_returns + rew_jax
        self.current_episode_len = self.current_episode_len + 1
        
        # When episodes end, update the logged returns/episode_len and reset current tracking
        done_mask = (dones_jax | truncations_jax).astype(jnp.float32)
        self.returns = done_mask * self.current_returns + (1.0 - done_mask) * self.returns
        self.episode_len = done_mask * self.current_episode_len + (1.0 - done_mask) * self.episode_len
        
        # Reset current tracking for done environments
        self.current_returns = (1.0 - done_mask) * self.current_returns
        self.current_episode_len = (1.0 - done_mask) * self.current_episode_len
        
        info_ret = {
            "time_outs": to_jax(truncations), 
            "observations": {"critic": to_jax(critic_obs) if critic_obs is not None else None},
            "log_info": {
                "return": self.returns,
                "episode_len": self.episode_len,
            }
        }
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        info_ret["observations"]["raw"] = {
            "obs": to_jax(obs),
            "critic_obs": to_jax(critic_obs) if critic_obs is not None else None,
        }

        return to_jax(obs), rew_jax, dones_jax, truncations_jax, info_ret

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )
