"""Wrapper to use offline datasets as environments."""
import torch
import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader
from src.maniskill_utils.maniskill_dataloader import ManiSkillTrajectoryDataset
from gymnasium import spaces

def make_space_from_obs(obs):
    if isinstance(obs, dict):
        return spaces.Dict({
            k: make_space_from_obs(v)
            for k, v in obs.items()
        })
    elif isinstance(obs, np.ndarray):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=obs.dtype
        )
    else:
        raise TypeError(type(obs))


class OfflineDatasetEnv:
    """Wrapper that mimics gymnasium.Env interface for offline datasets.
    
    Each dataloader item is a SINGLE STEP:
    (obs, action, terminated, truncated, success, reward, extra)
    
    Returns step format: (next_obs, reward, done, truncated, info)
    """

    def __init__(self, dataloader = ManiSkillTrajectoryDataset("/scratch/cluster/idutta/h5_files/trajectory.rgb.pd_joint_pos.physx_cpu.h5", load_count=100, success_only=False, device="cpu"), observation_space=None, action_space=None):
        """
        Args:
            dataloader: PyTorch DataLoader where each item is ONE step
                       yields: (obs, action, terminated, truncated, success, reward, extra)
            observation_space: gymnasium Space object (optional)
            action_space: gymnasium Space object (optional)
        """
        self.dataloader = dataloader
        first = self.dataloader[0]
        self.dataloader_iter = iter(self.dataloader)
        self.observation_space = make_space_from_obs(self.dataloader[0]["obs"])
        self.action_space = spaces.Box(
                            low=-1.0,   # or dataset metadata
                            high=1.0,
                            shape=self.dataloader[0]["action"].shape,
                            dtype=np.float32
                        ) 
        self.last_obs = None
        
    def _tensor_to_numpy(self, obj):
        """Recursively convert torch tensors to numpy."""
        if isinstance(obj, torch.Tensor):
            return obj.numpy() if not obj.is_cuda else obj.cpu().numpy()
        elif isinstance(obj, dict):
            return {k: self._tensor_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._tensor_to_numpy(item) for item in obj)
        return obj
    
    def _numpy_to_jax(self, obj):
        """Recursively convert numpy arrays to JAX arrays."""
        if isinstance(obj, np.ndarray):
            return jnp.asarray(obj)
        elif isinstance(obj, dict):
            return {k: self._numpy_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._numpy_to_jax(item) for item in obj)
        return obj
    
    def _to_scalar(self, val):
        """Convert tensor/array scalars to Python types."""
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                # For boolean tensors, ensure we return Python bool
                if val.dtype == torch.bool:
                    return bool(val.item())
                return val.item()
        elif isinstance(val, np.ndarray):
            if val.size == 1:
                # For boolean arrays, ensure we return Python bool
                if val.dtype == bool or val.dtype == np.bool_:
                    return bool(val.item())
                return val.item()
        elif isinstance(val, (bool, np.bool_)):
            return bool(val)
        return val
    
    def reset(self):
        """Reset to next episode by getting first step from dataloader.
        
        Returns:
            obs: Initial observation of the episode as JAX arrays (PyTreeNode compatible)
            info: Empty dict for compatibility
        """
        try:
            step_data = next(self.dataloader_iter)
            
            # Unpack single step
            if isinstance(step_data, tuple):
                obs, action, terminated, truncated, success, reward, extra = step_data
            else:
                obs = step_data['obs']
            
            # Convert torch tensors to numpy, then to JAX
            obs_numpy = self._tensor_to_numpy(obs)
            self.last_obs = self._numpy_to_jax(obs_numpy)
            
            return self.last_obs, {}
            
        except StopIteration:
            # Restart dataloader if exhausted
            self.dataloader_iter = iter(self.dataloader)
            return self.reset()
    
    def step(self, action):
        """Step by getting next step from dataloader.
        
        Args:
            action: Not used (comes from policy), data is from offline dataset
            
        Returns:
            next_obs: Observation at this step as JAX arrays (PyTreeNode compatible)
            reward: Reward for this step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated  
            info: Dict with additional info
        """
        try:
            step_data = next(self.dataloader_iter)
            
            # Unpack single step
            if isinstance(step_data, tuple):
                obs, action_data, terminated, truncated, success, reward, extra = step_data
            else:
                obs = step_data['obs']
                action_data = step_data.get('action')
                terminated = step_data.get('terminated', False)
                truncated = step_data.get('truncated', False)
                success = step_data.get('success', False)
                reward = step_data.get('reward', 0.0)
                extra = step_data.get('extra', {})
            
            # Convert torch tensors to numpy, then to JAX
            obs_numpy = self._tensor_to_numpy(obs)
            next_obs = self._numpy_to_jax(obs_numpy)
            
            step_reward = self._to_scalar(reward)
            step_done = self._to_scalar(terminated)
            step_truncated = self._to_scalar(truncated)
            step_success = self._to_scalar(success)
            
            # Build info dict
            info = {
                'success': step_success,
            }
            
            return next_obs, step_reward, step_done, step_truncated, info
            
        except StopIteration:
            # Episode ended, return done=True
            return self.last_obs, 0.0, True, False, {'success': False}