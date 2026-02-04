import h5py
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import mani_skill.envs
import gymnasium as gym
import cv2
import imageio
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from tensordict import TensorDict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split

@dataclass
class DemoConfig:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    normalize_observations: bool = True
    include_next_obs: bool = True
    filter_success_only: bool = True

from torch.utils.data import Dataset, DataLoader

class TensorDictDataset(Dataset):
    def __init__(self, tensordict):
        self.td = tensordict

    def __len__(self):
        return self.td.batch_size[0]

    def __getitem__(self, idx):
        item = {}
        for key in self.td.keys():
            val = self.td.get(key)
            # Take the idx-th element
            item[key] = val[idx]
        return item

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
            
        return all_trajectories

    def _convert_trajectory(self, traj_group: h5py.Group, episode_metadata: Dict) -> Optional[TensorDict]:
        actions = np.array(traj_group['actions'])
        terminated = np.array(traj_group['terminated'])
        truncated = np.array(traj_group['truncated'])
        T = len(actions)

        if T == 0:
            return None

        # Load observations
        if 'obs' in traj_group:
            observations = self._load_observations(traj_group['obs'])
        else:
            return None

        # Load rewards
        if "rewards" in traj_group and traj_group["rewards"] is not None:
            rewards = np.array(traj_group["rewards"])
        else:
            rewards = self._compute_dummy_rewards(traj_group, episode_metadata)

        # Load success and CUT trajectory at first success
        if 'success' in traj_group:
            success = np.array(traj_group['success'])
            if np.any(success):
                cut_idx = np.argmax(success) + 1
            else:
                cut_idx = T
        else:
            success = None
            cut_idx = T

        # Align everything properly (T-1 transitions)
        obs = observations[:cut_idx]
        act = actions[:cut_idx]
        rew = rewards[:cut_idx]
        term = terminated[:cut_idx]
        trunc = truncated[:cut_idx]

        if len(obs) < 2:
            return None

        traj_data = {
            'observations': torch.as_tensor(obs[:-1], dtype=self.config.dtype, device=self.config.device),
            'next_observations': torch.as_tensor(obs[1:], dtype=self.config.dtype, device=self.config.device),
            'actions': torch.as_tensor(act[:-1], dtype=self.config.dtype, device=self.config.device),
            'rewards': torch.as_tensor(rew[:-1], dtype=self.config.dtype, device=self.config.device).unsqueeze(-1),
            'dones': torch.as_tensor(term[:-1], dtype=torch.bool, device=self.config.device).unsqueeze(-1),
            'truncations': torch.as_tensor(trunc[:-1], dtype=torch.bool, device=self.config.device).unsqueeze(-1),
        }

        if success is not None:
            traj_data['success'] = torch.as_tensor(
                success[:cut_idx-1],
                dtype=torch.bool,
                device=self.config.device
            ).unsqueeze(-1)

        td = TensorDict(traj_data, batch_size=(len(obs) - 1,), device=self.config.device)
        return td.unsqueeze(0)


    def _load_observations(self, obs_group: h5py.Group) -> np.ndarray:
        # modified BC
        noise_std = 0.02     # standard deviation of Gaussian noise
        noise_indices = None # indices of observation dimensions to add noise to
        """Load observations from HDF5 group."""
        if isinstance(obs_group, h5py.Dataset):
            return np.array(obs_group)
        obs_data = []
        # modified BC
        robot_indices = []  # track which columns correspond to robot
        col_offset = 0
        for key in sorted(obs_group.keys()):  # iterates over available actors
            sub_group = obs_group[key]
            if key in ('agent', 'extra'):
                # print(key, sub_group.keys())
                for sub_key in sorted(sub_group.keys()):
                    data = np.array(sub_group[sub_key])
                    data_flat = data.reshape(data.shape[0], -1)
                    # mark agent columns for noise
                    if key == 'agent':
                        robot_indices.extend(range(col_offset, col_offset + data_flat.shape[1]))
                    col_offset += data_flat.shape[1]

                    obs_data.append(data_flat)

        obs_data = np.concatenate(obs_data, axis=1)
        # Add Gaussian noise to agent columns only
        if noise_std > 0.0 and len(robot_indices) > 0:
            obs_noisy = obs_data.copy()
            noise = np.random.randn(obs_noisy.shape[0], len(robot_indices)) * noise_std
            obs_noisy[:, robot_indices] += noise
            return obs_noisy

        return obs_data


    def _compute_dummy_rewards(self, traj_group: h5py.Group, episode_metadata: Dict) -> np.ndarray:
        # if 'rewards' in traj_group:
        #     return np.array(traj_group['rewards'])
        if 'success' in traj_group:
            success = np.array(traj_group['success'])
            rewards = np.zeros_like(success, dtype=np.float32)
            if np.any(success):
                rewards[np.argmax(success)] = 1.0
            return rewards
        T = len(traj_group['actions'])
        return np.full(T, -0.01, dtype=np.float32)

def load_demos_for_training(env_id: str,
                            bsize: int = 64,
                            demo_path: str = '/scratch/cluster/idutta/h5_files/trajectory.rgb.pd_joint_pos.physx_cpu.h5',
                            device: torch.device = torch.device("cpu"),
                            max_episodes: Optional[int] = None,
                            filter_success: bool = True) -> TensorDict:

    # initialize everything
    # demo_path = Path(f"{demo_dir}/{env_id}/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5")
    # demo_path = '/scratch/cluster/idutta/h5_files/trajectory.none.pd_joint_delta_pos.physx_cuda.h5'
    train_ratio = 0.8
    val_ratio = 0.2

    # create trajectory dataset
    config = DemoConfig(device=device, filter_success_only=filter_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, metadata = loader.load_demo_dataset(demo_path)
    print(f"Loaded {len(trajectories)} transitions from {demo_path}")
    # print(trajectories.batch_size[0])
    # print(trajectories['observations'][0].shape)

    # create dataset from these flattened trajectories
    # dataset = TensorDictDataset(trajectories)
    # total_len = len(dataset)
    # train_len = int(train_ratio * total_len)
    # val_len = total_len - train_len
    # train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # create dataset from non flattened trajectories
    # print(trajectories[0])
    num_traj = len(trajectories)
    train_traj = int(0.8 * num_traj)
    train_td = torch.cat(trajectories[:train_traj], dim=0)
    val_td = torch.cat(trajectories[train_traj:], dim=0)
    train_dataset = TensorDictDataset(train_td)
    val_dataset   = TensorDictDataset(val_td)

    train_loader = DataLoader(train_dataset,
        batch_size=bsize,
        shuffle=True,
        drop_last=True)
    val_loader = DataLoader(val_dataset,
        batch_size=bsize,
        shuffle=False,
        drop_last=False)
    obs_dim = trajectories[0]["observations"].shape[1]  # 25 for non flattened
    act_dim = trajectories[0]["actions"].shape[1]  # 8 for non flattened
    # find min and max action values
    # actions = trajectories["actions"]   # shape [N, 8]
    actions = torch.cat([td["actions"] for td in trajectories], dim=0) # for non flattened
    act_min = actions.min(dim=0).values   # [8]
    act_max = actions.max(dim=0).values   # [8]
    print(f"Created {len(train_loader)} and {len(val_loader)} dataset from {demo_path}")
    return train_loader, val_loader, obs_dim, act_dim, act_min, act_max

# result = load_demos_for_training("PushCube-v1", device=torch.device("cpu"), filter_success=True)

# Create video from dataset observations (render pre-recorded RGB or state images)
def render_trajectory_video(demo_path = '/scratch/cluster/idutta/h5_files/trajectory.rgb.pd_joint_pos.physx_cpu.h5', env_id = 'PushCube-v1', output_path = '/scratch/cluster/idutta/expert_videos', fps: int = 30):
    """
    Render videos by rendering pre-recorded RGB observations from dataset.
    Uses RGB images directly from H5 file sensor_data structure.
    
    Args:
        demo_path: Path to H5 trajectory file
        env_id: ManiSkill environment ID
        output_path: Directory to save MP4 videos
        fps: Frames per second for the video
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Load H5 file directly to get RGB observations
    with h5py.File(demo_path, 'r') as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith('traj_')])
        
        for idx, traj_key in enumerate(traj_keys):
            traj_group = f[traj_key]
            
            # Check if sensor_data with RGB is available
            if 'obs' in traj_group:
                obs_group = traj_group['obs']
                
                # Navigate to sensor_data structure
                if 'sensor_data' in obs_group:
                    sensor_data = obs_group['sensor_data']
                    
                    # Get the first sensor_uid that has RGB data
                    rgb_data = None
                    sensor_uid = None
                    
                    for uid in sensor_data.keys():
                        if 'rgb' in sensor_data[uid]:
                            sensor_uid = uid
                            rgb_data = np.array(sensor_data[uid]['rgb'])
                            break
                    
                    if rgb_data is not None:
                        video_path = os.path.join(output_path, f"{env_id}_traj_{idx}.mp4")
                        frames = []
                        
                        print(f"Rendering trajectory {idx} using sensor {sensor_uid}, RGB shape: {rgb_data.shape}")
                        
                        # Save RGB frames as video
                        for t in range(len(rgb_data)):
                            frame = rgb_data[t]
                            # Ensure frame is uint8
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                            frames.append(frame)
                        
                        if frames:
                            with imageio.get_writer(video_path, fps=fps) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                            print(f"Video saved to {video_path} ({len(frames)} frames)")
                        else:
                            print(f"No frames found for {video_path}")
                    else:
                        print(f"Warning: No RGB data found in trajectory {idx}")
                else:
                    print(f"Warning: No sensor_data found in trajectory {idx}")
    
    print(f"All trajectory videos saved to {output_path}")

# render_trajectory_video()

# In modified BC, we add a small Gaussian noise to the observations during training. 
# Reason:
# - In vanilla BC, the policy is trained to map observations coming from the expert distribution π to actions.
# - During execution, the policy sees states from its own trajectory distribution θ, which may differ slightly.
# - Small errors compound: a tiny deviation in one step moves the state out of the expert distribution. Without having seen these "near-miss" states, the next predicted action may be wrong, causing a catastrophic spiral away from the expert trajectory.

# Mathematical formulation:
# - Vanilla BC loss: L(θ) = E_(s,a)~D [-log π_θ(a|s)]
# - With state noise s' = s + ε, ε ~ N(0, σ^2):
#   L_noise(θ) = E_(s,a)~D, ε~N(0,σ^2) [-log π_θ(a | s + ε)]

# - Using a second-order Taylor expansion around ε=0:
#   L_noise ≈ L + 1/2 * Tr(Σ * E_s[H_s(-log π_θ(a|s))])
#   where H_s is the Hessian of -log π_θ w.r.t. the state s.

# Interpretation:
# - The second term penalizes high curvature in the log-probability surface.
# - Intuitively, it forces the policy to be smooth: small deviations in state (drifts) do not drastically reduce the probability of taking the correct expert action.
# - Mathematically, the optimizer minimizes both the loss and the local gradients of π_θ(s), effectively reducing the Lipschitz constant of the policy and increasing robustness.

# Adding noise smoothens the policy but does not solve the compounding error problem entirely. As it creates a local bound around the expert states, that still does not help once the policy sees states it has never seen before rather the states that are outside the local bound of this noise.
# This problem once startes accumulating can compound over time. 
# There is one more problem, the std of the policy is 0.01 - 0.05 which is very loss which is causing the policy to become determininstic and not explore bad options. The policy is getting biased a towards ction dimensions with this std value
# Action mean per dim: [-1.1850e-03,  5.9062e-01,  4.3090e-04, -1.9659e+00, -7.7145e-04, 2.5577e+00,  7.8363e-01, -9.9997e-01]
# Action std  per dim: [0.0668, 0.1995, 0.0240, 0.2744, 0.0302, 0.1747, 0.0909, 0.0042]
# Mean action L2 norm: 3.532094717025757 and Predicted action L2 norm ~ 2.2
# Added a regularizer using the log std to make the policy more stochastic and increase the min_std parameter. Also added an auxiliary MSE loss between the expert action means and the means of the multivariate distribution, in addition to the NLL loss.
# This caused the rewards to increase but the actions became way too large and unstable.