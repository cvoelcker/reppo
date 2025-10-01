import h5py
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tensordict import TensorDict
import gymnasium as gym
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DemoConfig:
    """Configuration for demo loading."""
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    normalize_observations: bool = True
    include_next_obs: bool = True
    filter_success_only: bool = True


class ManiSkillDemoLoader:
    
    def __init__(self, config: DemoConfig):
        self.config = config
        
    def load_demo_dataset(
        self, 
        trajectory_path: Union[str, Path],
    ) -> Tuple[TensorDict, Dict]:
        trajectory_path = Path(trajectory_path)
        metadata_path = trajectory_path.with_suffix('.json')
            
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        trajectories = self._load_trajectories(trajectory_path, metadata)
        
        return trajectories, metadata
    
    def _load_trajectories(
        self, 
        trajectory_path: Path, 
        metadata: Dict
    ) -> TensorDict:
        all_trajectories = []
        with h5py.File(trajectory_path, 'r') as f:
            traj_keys = [key for key in f.keys() if key.startswith('traj_')]
            for traj_key in traj_keys:
                traj_group = f[traj_key]
                episode_id = int(traj_key.split('_')[1])
                episode_metadata = {}
                if metadata and 'episodes' in metadata:
                    episode_metadata = next(
                        (ep for ep in metadata['episodes'] if ep['episode_id'] == episode_id),
                        {}
                    )

                if self.config.filter_success_only and 'success' in traj_group:
                    success = traj_group['success'][:]
                    if not np.any(success):
                        continue
                      
                # single trajectory
                traj_tensordict = self._convert_trajectory(traj_group, episode_metadata)
                if traj_tensordict is not None:
                    td = traj_tensordict.float().flatten(0, 1).detach()
                    all_trajectories.append(td)
        if not all_trajectories:
            raise ValueError("No valid trajectories found in the dataset")
          
        # Stack all trajectories
        return torch.cat(all_trajectories, dim=0)
    
    def _convert_trajectory(
        self, 
        traj_group: h5py.Group, 
        episode_metadata: Dict
    ) -> Optional[TensorDict]:
        """Convert a SINGLE trajectory to TensorDict."""
        try:
            actions = np.array(traj_group['actions'])
            terminated = np.array(traj_group['terminated'])
            truncated = np.array(traj_group['truncated'])
            rewards = np.array(traj_group['rewards']) if 'rewards' in traj_group else None
            
            #  trajectory length
            T = len(actions)
            if T == 0:
                return None
            
            observations = None
            if 'obs' in traj_group:
                observations = self._load_observations(traj_group['obs'])
            else:
                # dummy observations
                observations = np.zeros((T + 1, 1))
            
            rewards = self._compute_dummy_rewards(traj_group, episode_metadata)
            
            success = None
            if 'success' in traj_group:
                success = np.array(traj_group['success'])
            
            trajectory_data = {
                'observations': torch.as_tensor(
                    observations[:-1], 
                    dtype=self.config.dtype, 
                    device=self.config.device
                ),
                'next_observations': torch.as_tensor(
                    observations[1:], 
                    dtype=self.config.dtype, 
                    device=self.config.device
                ),
                'actions': torch.as_tensor(
                    actions, 
                    dtype=self.config.dtype, 
                    device=self.config.device
                ),
                'rewards': torch.as_tensor(
                    rewards, 
                    dtype=self.config.dtype, 
                    device=self.config.device
                ).unsqueeze(-1),
                'dones': torch.as_tensor(
                    terminated, 
                    dtype=torch.bool, 
                    device=self.config.device
                ).unsqueeze(-1),
                'truncations': torch.as_tensor(
                    truncated, 
                    dtype=torch.bool, 
                    device=self.config.device
                ).unsqueeze(-1),
            }
            
            if success is not None:
                trajectory_data['success'] = torch.as_tensor(
                    success, 
                    dtype=torch.bool, 
                    device=self.config.device
                ).unsqueeze(-1)
            
            tensordict = TensorDict(
                trajectory_data,
                batch_size=(T,),
                device=self.config.device
            )
            
            return tensordict.unsqueeze(0)  # Add batch dimension?
            
        except Exception as e:
            print(f"Error converting trajectory: {e}")
            return None
    
    def _load_observations(self, obs_group: h5py.Group) -> np.ndarray:
        """Load observations from HDF5 group."""
        if isinstance(obs_group, h5py.Dataset):
            return np.array(obs_group)
        else:
            # flatten or concatenate
            obs_data = []
            for key in sorted(obs_group.keys()):
                data = np.array(obs_group[key])
                if data.ndim == 1:
                    obs_data.append(data.reshape(-1, 1))
                else:
                    obs_data.append(data.reshape(data.shape[0], -1))
            
            if obs_data:
                return np.concatenate(obs_data, axis=1)
            else:
                return np.zeros((1, 1))

    def _compute_dummy_rewards(
        self, 
        traj_group: h5py.Group, 
        episode_metadata: Dict
    ) -> np.ndarray:
        """Compute or extract rewards from trajectory."""
        
        if 'rewards' in traj_group:
            return np.array(traj_group['rewards'])
        
        # If no rewards, compute from success
        if 'success' in traj_group:
            success = np.array(traj_group['success'])
            # 1.0 when task succeeds, 0.0 otherwise
            rewards = success.astype(np.float32)
            # Give final reward only at the end
            final_reward = np.zeros_like(rewards)
            if np.any(success):
                # Find first success and give reward there
                success_idx = np.where(success)[0][0]
                final_reward[success_idx] = 1.0
            return final_reward
        
        # If no success information, give small negative reward to encourage efficiency
        T = len(traj_group['actions'])
        return np.full(T, -0.01, dtype=np.float32)


def create_demo_dataset(
    demo_paths: List[Union[str, Path]],
    config: DemoConfig,
    env_id: Optional[str] = None
) -> TensorDict:
    loader = ManiSkillDemoLoader(config)
    all_trajectories = []
    
    for demo_path in demo_paths:
        try:
            trajectories, metadata = loader.load_demo_dataset(demo_path)
            # Validate environment ID if provided
            if env_id and metadata.get('env_info', {}).get('env_id') != env_id:
                print(f"Warning: Expected env_id {env_id}, got {metadata.get('env_info', {}).get('env_id')}")
            print(f"Validated env_id for {demo_path}")
            all_trajectories.append(trajectories)
            print(f"Loaded {len(trajectories)} trajectories from {demo_path}")
            
        except Exception as e:
            print(f"Error loading {demo_path}: {e}")
            continue
    
    if not all_trajectories:
        raise ValueError("No trajectories could be loaded")
    
    combined_dataset = torch.cat(all_trajectories, dim=0)
    print(f"Combined dataset shape: {combined_dataset.shape}")
    
    return combined_dataset

def load_demos_for_training(
    env_id: str,
    demo_dir: Union[str, Path],
    device: torch.device = torch.device("cpu"),
    max_episodes: Optional[int] = None,
    filter_success: bool = True
) -> TensorDict:
    demo_dir = Path(demo_dir)
    #  <demo_dir>/<env_id>/rl/trajectory.*.h5
    # modes: (ee-delta-pos, ee-delta-pose, joint-delta-pos)
    # the one the simulator
    subdir = demo_dir / env_id / "rl"
    if subdir.exists():
      # joint-delta-pos
      trajectory_files = subdir / "trajectory.none.pd_joint_delta_pos.physx_cuda.h5"

    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory files found for {env_id} in {demo_dir}")
    
    config = DemoConfig(
        device=device,
        filter_success_only=filter_success
    )
    # remove [] is more than one file
    return create_demo_dataset([trajectory_files], config, env_id)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        demo_data = load_demos_for_training(
            env_id="RollBall-v1",
            demo_dir="/home/tajiksh1/.maniskill/demos",
            device=device,
            filter_success=True
        )
        print(f"Loaded demo dataset with shape: {demo_data.shape}")
        print(f"Keys: {demo_data.keys()}")
    except Exception as e:
        print(f"Error: {e}")
