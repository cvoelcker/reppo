import h5py
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from tensordict import TensorDict
from dataclasses import dataclass

@dataclass
class DemoConfig:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    normalize_observations: bool = True
    include_next_obs: bool = True
    filter_success_only: bool = True

class ManiSkillDemoLoader:
    def __init__(self, config: DemoConfig, env_id: str):
        self.config = config

    def load_demo_dataset(self, trajectory_path: Union[str, Path]) -> Tuple[TensorDict, Dict]:
        trajectory_path = Path(trajectory_path)
        metadata_path = trajectory_path.with_suffix('.json')
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        trajectories = self._load_trajectories(trajectory_path, metadata)
        return trajectories, metadata

    def _load_trajectories(self, trajectory_path: Path, metadata: Dict) -> TensorDict:
        all_trajectories = []
        with h5py.File(trajectory_path, 'r') as f:
            traj_keys = [key for key in f.keys() if key.startswith('traj_')]
            for traj_key in traj_keys:
                traj_group = f[traj_key]
                episode_id = int(traj_key.split('_')[1])
                episode_metadata = next(
                    (ep for ep in metadata.get('episodes', []) if ep['episode_id'] == episode_id),
                    {}
                )

                # Filter out failed trajectories if needed
                if self.config.filter_success_only and 'success' in traj_group:
                    success = traj_group['success'][:]
                    if not np.any(success):
                        continue
                
                td = self._convert_trajectory(traj_group, episode_metadata)
                if td is not None:
                    all_trajectories.append(td.float().flatten(0, 1).detach())
        
        if not all_trajectories:
            raise ValueError("No valid trajectories found in the dataset")
        
        return torch.cat(all_trajectories, dim=0)

    def _convert_trajectory(self, traj_group: h5py.Group, episode_metadata: Dict) -> Optional[TensorDict]:
        try:
            actions = np.array(traj_group['actions'])
            terminated = np.array(traj_group['terminated'])
            truncated = np.array(traj_group['truncated'])
            T = len(actions)
            if T == 0:
                return None

            if 'obs' in traj_group:
                observations = self._load_observations(traj_group['obs'])
            elif 'env_states' in traj_group:
                observations = self._load_observations(traj_group['env_states'])

            rewards = self._compute_dummy_rewards(traj_group, episode_metadata)
            success = np.array(traj_group['success']) if 'success' in traj_group else None

            traj_data = {
                'observations': torch.as_tensor(observations[:-1], dtype=self.config.dtype, device=self.config.device),
                'next_observations': torch.as_tensor(observations[1:], dtype=self.config.dtype, device=self.config.device),
                'actions': torch.as_tensor(actions, dtype=self.config.dtype, device=self.config.device),
                'rewards': torch.as_tensor(rewards, dtype=self.config.dtype, device=self.config.device).unsqueeze(-1),
                'dones': torch.as_tensor(terminated, dtype=torch.bool, device=self.config.device).unsqueeze(-1),
                'truncations': torch.as_tensor(truncated, dtype=torch.bool, device=self.config.device).unsqueeze(-1),
            }

            if success is not None:
                traj_data['success'] = torch.as_tensor(success, dtype=torch.bool, device=self.config.device).unsqueeze(-1)

            td = TensorDict(traj_data, batch_size=(T,), device=self.config.device)
            return td.unsqueeze(0)
        except Exception as e:
            print(f"Error converting trajectory: {e}")
            return None

    def _load_observations(self, obs_group: h5py.Group) -> np.ndarray:
        """Load observations from HDF5 group."""
        if isinstance(obs_group, h5py.Dataset):
            return np.array(obs_group)
        obs_data = []
        for key in sorted(obs_group.keys()):  # iterates over available actors
            sub_group = obs_group[key]
            for sub_key in sorted(sub_group.keys()):
                data = np.array(sub_group[sub_key])
                obs_data.append(data.reshape(data.shape[0], -1))
        return np.concatenate(obs_data, axis=1)


    def _compute_dummy_rewards(self, traj_group: h5py.Group, episode_metadata: Dict) -> np.ndarray:
        if 'rewards' in traj_group:
            return np.array(traj_group['rewards'])
        if 'success' in traj_group:
            success = np.array(traj_group['success'])
            rewards = np.zeros_like(success, dtype=np.float32)
            if np.any(success):
                rewards[np.argmax(success)] = 1.0
            return rewards
        T = len(traj_group['actions'])
        return np.full(T, -0.01, dtype=np.float32)

def load_demos_for_training(env_id: str, demo_dir: Union[str, Path],
                            device: torch.device = torch.device("cpu"),
                            max_episodes: Optional[int] = None,
                            filter_success: bool = True) -> TensorDict:
    demo_path = Path(f"{demo_dir}/{env_id}/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5")
    config = DemoConfig(device=device, filter_success_only=filter_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, metadata = loader.load_demo_dataset(demo_path)
    print(f"Loaded {len(trajectories)} transitions from {demo_path}")
    return trajectories
