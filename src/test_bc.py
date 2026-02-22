import torch
import os
import mani_skill.envs
import gymnasium as gym
import imageio
import numpy as np
from omegaconf import OmegaConf
from collections import defaultdict
from src.network_utils.torch_models import Actor
import hydra
import wandb
from pathlib import Path
import h5py
from datetime import datetime
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader

def make_eval_env(env_id, control_mode="pd_joint_pos", seed=0, render=True):
    env = gym.make(
        env_id,
        obs_mode="state_dict",          # MUST match training obs
        control_mode=control_mode,
        render_mode="rgb_array" if render else None,
        reward_mode = "normalized_dense", # Instead of just a sparse "success/fail" signal at the end, it gives normalized or scaled rewards (e.g., [−α, α] or [1−β, 1+β]) at each time step, indicating how well the agent is doing.
        max_episode_steps=100
    )
    env.reset(seed=seed)
    return env

def replay_expert_and_measure_mse(cfg, save_dir, device='cpu'):
    """
    Replay expert trajectories in a real environment using raw dataset actions.
    Records videos of the expert replays.
    """

    config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
    loader = ManiSkillDemoLoader(config, cfg.env.name)
    trajectories, metadata = loader.load_demo_dataset(cfg.env.demo.demo_path)
    
    # Create subdirectory for expert replay videos (one level above save_dir)
    expert_video_dir = Path(save_dir).parent / "expert_replays"
    expert_video_dir.mkdir(parents=True, exist_ok=True)
    
    demo_obs_keys = get_demo_obs_keys(cfg.env.demo.demo_path)
    
    for traj_idx, traj in enumerate(trajectories[:200]):
        obs_data = traj['observations'] 
        expert_actions = traj['actions']
        
        # Create environment for this replay
        env = make_eval_env(cfg.env.name, control_mode=cfg.env.get('control_mode'), seed=traj_idx, render=True)
        
        frames = []
        traj_reward = 0.0
        traj_success = False
        
        print(f"\n=== REPLAYING TRAJECTORY {traj_idx} ===")
        print(f"Trajectory length: {len(expert_actions)} steps")
        
        obs, _ = env.reset()
        
        # Capture initial frame after reset
        frame = env.render()
        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            frames.append(frame)
        
        # Replay trajectory using raw expert actions
        for t in range(len(expert_actions)):
            expert_action = expert_actions[t].cpu().numpy()
            
            # ManiSkill will automatically normalize this action based on controller config
            # (normalize_action=True by default for pd_joint_delta_pos)
            # Step environment with raw expert action
            obs, reward, terminated, truncated, info = env.step(expert_action)
            
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            traj_reward += reward
            traj_success |= info.get("success", False)
            
            # Capture frame after step
            frame = env.render()
            if frame is not None:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                frames.append(frame)
            
            if terminated or truncated:
                break
        
        if frames:
            video_path = str(expert_video_dir / f"traj_{traj_idx:03d}.mp4")
            frames_stacked = np.stack(frames, axis=0)
            imageio.mimsave(video_path, frames_stacked, fps=30)
            print(f"Saved expert replay video: {video_path}")
        
        print(f"Expert replay - Reward: {traj_reward:.3f}, Success: {traj_success}, Steps: {len(frames)}")
        print(f"=" * 50)
        
        env.close()

def get_demo_obs_keys(demo_path):
    """Extract which observation keys are actually in the demo file."""
    import h5py
    
    with h5py.File(demo_path, 'r') as f:
        traj_group = f['traj_0']
        obs_group = traj_group['obs']
        
        demo_keys = {'agent': [], 'extra': []}
        for key in sorted(obs_group.keys()):
            if key in ('agent', 'extra'):
                sub_group = obs_group[key]
                if hasattr(sub_group, 'keys'):
                    demo_keys[key] = sorted(sub_group.keys())
        
        return demo_keys

def flatten_obs(obs_dict, env, demo_obs_keys):
    # Flatten the observation dictionary to match the demo data format
    # Only include keys that were actually in the demo file
    obs_list = []
    for key in sorted(obs_dict.keys()):
        if key in ('agent', 'extra') and key in demo_obs_keys:
            sub_group = obs_dict[key]
            for sub_key in sorted(sub_group.keys()):
                # Only include keys that were in the demo
                if sub_key in demo_obs_keys[key]:
                    data = sub_group[sub_key]
                    if isinstance(data, np.ndarray):
                        data_flat = data.reshape(data.shape[0], -1) if len(data.shape) > 1 else data.reshape(1, -1)
                    else:
                        data_array = np.asarray(data)
                        data_flat = data_array.reshape(1, -1) if data_array.ndim == 1 else data_array.reshape(data_array.shape[0], -1)
                    obs_list.append(data_flat)
    
    # Add env_states/actors data
    # if hasattr(env.unwrapped, 'get_state_dict'):
    #     state = env.unwrapped.get_state_dict()
    #     if 'actors' in state:
    #         for actor_name in sorted(state['actors'].keys()):
    #             actor_data = state['actors'][actor_name]
    #             if isinstance(actor_data, np.ndarray):
    #                 data_flat = actor_data.reshape(actor_data.shape[0], -1) if len(actor_data.shape) > 1 else actor_data.reshape(1, -1)
    #             else:
    #                 data_array = np.asarray(actor_data.cpu()) if hasattr(actor_data, 'cpu') else np.asarray(actor_data)
    #                 data_flat = data_array.reshape(1, -1) if data_array.ndim == 1 else data_array.reshape(data_array.shape[0], -1)
    #             obs_list.append(data_flat)
    
    if not obs_list:
        raise ValueError(f"No observation data found in obs_dict with keys: {obs_dict.keys()}")
    
    result = np.concatenate(obs_list, axis=1)
    return result

def test(cfg, env_id=None, model_path=None, demo_path=None):
    # Initialize wandb
    wandb.init(
        config=dict(cfg),
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        name=f"bc_eval_no_clamp_linear_scale_no_envstates_{cfg.env.name}",
        mode=cfg.logging.mode
    )
    
    # Use cfg from Hydra if provided, otherwise load it
    if cfg is None:
        config_path = Path(__file__).parent.parent / "config" / "default" / "reppo_maniskill.yaml"
        cfg = OmegaConf.load(str(config_path))
    
    # Override env_id and demo_path if provided
    if env_id:
        cfg.env.name = env_id
    if demo_path:
        cfg.env.demo.demo_path = demo_path
    
    env_id = cfg.env.name
    demo_path = cfg.env.demo.demo_path

    # Get demo observation keys to know which fields to extract
    demo_obs_keys = get_demo_obs_keys(demo_path)
    print(f"Demo observation keys: {demo_obs_keys}")

    # Load actor
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Dynamically determine observation and action dimensions from demo data
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories_for_dims, _ = loader.load_demo_dataset(demo_path)
    n_obs = trajectories_for_dims[0]["observations"].shape[1]
    n_act = trajectories_for_dims[0]["actions"].shape[1]
    
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        ent_start=cfg.algorithm.ent_start,
        kl_start=cfg.algorithm.kl_start,
        hidden_dim=cfg.algorithm.actor_hidden_dim,
        use_norm=cfg.algorithm.use_actor_norm,
        layers=cfg.algorithm.num_actor_layers,
        min_std=cfg.algorithm.actor_min_std,
        device=device,
    )
    print(f"Actor created with n_obs={n_obs}, n_act={n_act}")
    
    # Load trained weights
    if model_path is None:
        # Prefer per-env best model path: outputs/<date>/<time>/<env_id>/saved_models/<timestamp>/bc_model_actor_best.pt
        outputs_dir = Path(__file__).parent.parent / "outputs"
        if outputs_dir.exists():
            best_candidates = list(
                outputs_dir.glob(f"*/*/{env_id}/saved_models/*/bc_model_actor_best.pt")
            )
            if best_candidates:
                model_path = str(max(best_candidates, key=lambda p: p.stat().st_mtime))
        else:
            raise ValueError(f"Outputs directory not found: {outputs_dir}")
    
    # Ensure model_path is absolute or properly relative
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    actor.load_state_dict(torch.load(str(model_path), map_location=device))
    actor.eval()
    
    for param in actor.parameters():
        param.requires_grad = False

    # Create a temporary env to get action bounds
    temp_env = gym.make(env_id, control_mode=cfg.env.get('control_mode'), obs_mode="state_dict")
    env_low = torch.from_numpy(temp_env.action_space.low).float()
    env_high = torch.from_numpy(temp_env.action_space.high).float()
    temp_env.close()
    
    print(f"\n=== ENV ACTION SPACE INFO ===")
    print(f"Action bounds - Low: {env_low.numpy()}")
    print(f"Action bounds - High: {env_high.numpy()}")
    print(f"Action space range: {(env_high - env_low).numpy()}")
    print(f"Action space center: {((env_high + env_low) / 2).numpy()}")
    print(f"=" * 50)
    
    # Collect all actions from dataset and compute bounds with safety margin
    all_dataset_actions = torch.cat([traj['actions'] for traj in trajectories_for_dims], dim=0)
    data_low = all_dataset_actions.min(dim=0).values
    data_high = all_dataset_actions.max(dim=0).values
    
    # Add 10% safety margin to bounds (same as training)
    margin = 0.1 * (data_high - data_low)
    low_with_margin = data_low - margin
    high_with_margin = data_high + margin
    
    # Ensure bounds are at least [-1, 1] in each dimension
    dataset_low = torch.min(low_with_margin, torch.full_like(low_with_margin, -1.0))
    dataset_high = torch.max(high_with_margin, torch.full_like(high_with_margin, 1.0))
    
    print(f"\n=== OFFLINE DATASET ACTION BOUNDS (with 10% safety margin) ===")
    print(f"Dataset Action bounds - Low: {dataset_low.numpy()}")
    print(f"Dataset Action bounds - High: {dataset_high.numpy()}")
    print(f"Dataset space range: {(dataset_high - dataset_low).numpy()}")
    print(f"Dataset space center: {((dataset_high + dataset_low) / 2).numpy()}")
    print(f"=" * 50)

    num_episodes=500
    max_steps=100
    
    # Save videos in the same directory as the model
    model_parent_dir = Path(model_path).parent
    save_dir = model_parent_dir / "eval_videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    # save stats
    stats = defaultdict(list)

    for ep in range(num_episodes):
        env = make_eval_env(env_id, control_mode=cfg.env.get('control_mode'), seed=ep, render=True)
        obs, _ = env.reset()
        obs_tensor = torch.as_tensor(flatten_obs(obs, env, demo_obs_keys), dtype=torch.float32, device=device)

        frames = []
        ep_reward = 0.0
        success = False
        out_of_bounds_count = 0

        for t in range(max_steps):

            with torch.no_grad():
                pi, tanh_mean, _, _, log_std, mean = actor(obs_tensor.to(device))
                
                # Sample action from the distribution (in [-1, 1] normalized space)
                normalized_action = pi.sample().squeeze(0).cpu()  # Keep as tensor for computation
                
                # Rescale from [-1, 1] to original per-dimension dataset bounds
                # Inverse of training normalization: action = (normalized + 1) * (high - low) / 2 + low
                action = (normalized_action + 1.0) * (dataset_high - dataset_low) / 2.0 + dataset_low
                action = action.numpy()
            
            # Use rescaled action for environment
            # ManiSkill's controller will handle normalization internally
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            success |= info.get("success", False)

            frame = env.render()
            if frame is not None:
                frame = frame.cpu().numpy()
                frames.append(frame.squeeze(0))

            if terminated or truncated:
                break
            
            obs_tensor = torch.as_tensor(flatten_obs(obs, env, demo_obs_keys), dtype=torch.float32, device=device)

        # save video
        video_path = str(save_dir / f"ep_{ep:03d}.mp4")
        frames = np.stack(frames, axis=0)
        imageio.mimsave(video_path, frames, fps=30)

        # Log stats
        stats["reward"].append(ep_reward)
        stats["success"].append(float(success))
        stats["length"].append(t + 1)
        stats["out_of_bounds"].append(out_of_bounds_count)
        
        # Log to wandb per episode
        wandb.log({
            "eval/reward": ep_reward,
            "eval/success": float(success),
            "eval/episode_length": t + 1,
            "eval/out_of_bounds_count": out_of_bounds_count,
        }, step=ep)

        print(
            f"EP {ep:02d} | reward {ep_reward.cpu().item():.2f} | "
            f"success {success} | steps {t+1} | out_of_bounds {out_of_bounds_count}"
        )

        env.close()

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Average reward: {np.mean(stats['reward']):.3f}")
    print(f"Success rate: {np.mean(stats['success']):.1%}")
    print(f"Average episode length: {np.mean(stats['length']):.1f}")
    print(f"Average out-of-bounds steps per episode: {np.mean(stats['out_of_bounds']):.1f}")
    print(f"=" * 50)

    wandb.log({
        "eval/avg_reward": np.mean(stats['reward']),
        "eval/success_rate": np.mean(stats['success']),
        "eval/avg_episode_length": np.mean(stats['length']),
        "eval/avg_out_of_bounds": np.mean(stats['out_of_bounds']),
    })
    
    wandb.finish()
    
    # Replay expert trajectories in environment and record videos
    # replay_expert_and_measure_mse(
    #     cfg,
    #     save_dir,
    #     device=device,
    # )

@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    model_path = OmegaConf.select(cfg, "model_path")
    test(cfg=cfg, env_id=None, model_path=model_path, demo_path=cfg.env.demo.demo_path)

if __name__ == "__main__":
    main()