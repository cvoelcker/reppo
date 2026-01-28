import torch
import os
import mani_skill.envs
import gymnasium as gym
import imageio
import numpy as np
from omegaconf import OmegaConf
from collections import defaultdict
from src.network_utils.torch_models import Actor

def denormalize_action(normalized_action, low, high):
    """
    Denormalize actions from [-1, 1] back to original action space [low, high].
    This is the inverse of: normalized = 2.0 * (action - low) / (high - low) - 1.0
    """
    # Inverse formula: action = (normalized + 1.0) * (high - low) / 2.0 + low
    denormalized = (normalized_action + 1.0) * (high - low) / 2.0 + low
    return denormalized

def make_eval_env(env_id, seed=0, render=True):
    env = gym.make(
        env_id,
        obs_mode="state_dict",          # MUST match training obs
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if render else None,
        reward_mode = "normalized_dense" # Instead of just a sparse "success/fail" signal at the end, it gives normalized or scaled rewards (e.g., [−α, α] or [1−β, 1+β]) at each time step, indicating how well the agent is doing.
    )
    env.reset(seed=seed)
    return env

def replay_expert_and_measure_mse(actor, dataset_low, dataset_high, env_low, env_high, demo_path="/scratch/cluster/idutta/h5_files/trajectory.rgb.pd_joint_pos.physx_cpu.h5", device='cpu'):
    """
    Replay expert states from H5 file and measure MSE between policy and expert actions.
    Policy outputs are normalized by DATASET bounds, so denormalize using dataset bounds, then clip to env bounds.
    """
    import h5py
    from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
    
    # Load expert demonstrations
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
    loader = ManiSkillDemoLoader(config, 'PushCube-v1')
    trajectories, metadata = loader.load_demo_dataset(demo_path)
    
    # Run the policy on the dataset to log stats and MSE
    for traj_idx, traj in enumerate(trajectories[:3]):  # Sample first 3 trajectories
        obs = traj['observations']  # [T, 25]
        expert_actions = traj['actions']  # [T, 8]
        mse_list = []
        policy_action_list = []
        
        # Compute overall expert action std across entire trajectory (not per timestep)
        expert_action_std_overall = expert_actions.std(dim=0).mean().item()
        
        for t in range(len(obs)):
            obs_t = obs[t:t+1].to(device)  # [1, 25]
            expert_action_t = expert_actions[t:t+1].to(device)  # [1, 8]
            
            with torch.no_grad():
                pi, tanh_mean, _, _, log_std, mean = actor(obs_t)
                
                # Sample action from the distribution (in normalized [-1, 1] space)
                sampled_action_normalized = pi.sample().squeeze(0).cpu().numpy()
                
                # Denormalize using DATASET bounds (policy was trained on dataset-normalized actions)
                policy_action = denormalize_action(sampled_action_normalized, dataset_low.numpy(), dataset_high.numpy())
                
                # Clip to ENV bounds before feeding to environment
                policy_action = np.clip(policy_action, env_low.numpy(), env_high.numpy())
                
                # Store for std computation
                policy_action_list.append(policy_action)
                
                # MSE between policy and expert
                mse = np.mean((policy_action - expert_action_t.squeeze(0).cpu().numpy())**2)
                mse_list.append(mse)
        
        # Compute std of sampled policy actions across trajectory
        policy_action_array = np.array(policy_action_list)
        policy_action_std_overall = policy_action_array.std(axis=0).mean() if len(policy_action_list) > 1 else 0.0
        
        # Print stats per trajectory
        print(f"\n=== TRAJECTORY {traj_idx} EXPERT REPLAY MSE ===")
        print(f"Mean MSE (policy vs expert actions): {np.mean(mse_list):.4f}")
        print(f"Std MSE: {np.std(mse_list):.4f}")
        print(f"Policy action std (sampled): {policy_action_std_overall:.4f}")
        print(f"Expert action std: {expert_action_std_overall:.4f}")
        print(f"Std ratio (policy/expert): {policy_action_std_overall / (expert_action_std_overall + 1e-6):.2f}x")
        print(f"=" * 50)

def flatten_obs(obs_dict):
    # flatten the observation dictionary into a single array.
    obs_list = []
    for key in sorted(obs_dict.keys()):
        if key in ['agent', 'extra']:
            subkeys = obs_dict[key].keys()
            for subkey in subkeys:
                if subkey in ['qpos', 'qvel', 'tcp_pose']:
                    obs_list.append(obs_dict[key][subkey].reshape(obs_dict[key][subkey].shape[0], -1))
    return np.concatenate(obs_list, axis=1)

def test(env_id = 'PushCube-v1', cfg_path = "../reppo/config/algorithm/reppo.yaml",  model_path = "../reppo/bc_utils/bc_model_actor_20260127_195206_180.pth"):
    # Load config
    cfg = OmegaConf.load(cfg_path)

    # Load actor
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_obs = 25
    n_act = 8
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        ent_start=cfg.ent_start,
        kl_start=cfg.kl_start,
        hidden_dim=cfg.actor_hidden_dim,
        use_norm=cfg.use_actor_norm,
        layers=cfg.num_actor_layers,
        min_std=cfg.actor_min_std,
        device=device,
    )
    
    # Register learnable dimension weights parameter (COMMENTED OUT - not helping)
    # dim_weights_param = torch.nn.Parameter(torch.ones(n_act, device=device))
    # actor.register_parameter('dim_weights', dim_weights_param)
    
    # Load trained weights
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    
    # Stop gradients for all parameters during evaluation
    for param in actor.parameters():
        param.requires_grad = False

    # Create a temporary env to get action bounds
    temp_env = gym.make(env_id, obs_mode="state_dict", control_mode="pd_joint_pos")
    env_low = torch.from_numpy(temp_env.action_space.low).float()
    env_high = torch.from_numpy(temp_env.action_space.high).float()
    temp_env.close()
    
    print(f"\n=== ENV ACTION SPACE INFO ===")
    print(f"Action bounds - Low: {env_low.numpy()}")
    print(f"Action bounds - High: {env_high.numpy()}")
    print(f"Action space range: {(env_high - env_low).numpy()}")
    print(f"Action space center: {((env_high + env_low) / 2).numpy()}")
    print(f"=" * 50)
    
    # Load offline dataset and check action bounds
    from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories_for_bounds, _ = loader.load_demo_dataset(
        trajectory_path="/scratch/cluster/idutta/h5_files/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    )
    
    # Collect all actions from dataset
    all_dataset_actions = torch.cat([traj['actions'] for traj in trajectories_for_bounds], dim=0)
    dataset_low = all_dataset_actions.min(dim=0).values
    dataset_high = all_dataset_actions.max(dim=0).values
    
    print(f"\n=== OFFLINE DATASET ACTION BOUNDS ===")
    print(f"Dataset Action bounds - Low: {dataset_low.numpy()}")
    print(f"Dataset Action bounds - High: {dataset_high.numpy()}")
    print(f"Dataset space range: {(dataset_high - dataset_low).numpy()}")
    print(f"Dataset space center: {((dataset_high + dataset_low) / 2).numpy()}")
    print(f"=" * 50)
    
    # Compare env bounds with dataset bounds
    print(f"\n=== COMPARISON: ENV vs DATASET ===")
    print(f"Action dim | Env Low  | Dataset Low | Match?")
    for i in range(len(env_low)):
        match = "✓" if torch.isclose(env_low[i], dataset_low[i], atol=0.01) else "✗"
        print(f"    {i}    | {env_low[i]:7.3f} | {dataset_low[i]:10.3f} | {match}")
    
    print(f"\nAction dim | Env High | Dataset High | Match?")
    for i in range(len(env_high)):
        match = "✓" if torch.isclose(env_high[i], dataset_high[i], atol=0.01) else "✗"
        print(f"    {i}    | {env_high[i]:7.3f} | {dataset_high[i]:12.3f} | {match}")
    print(f"=" * 50)

    # define configs
    num_episodes=300
    max_steps=1000
    parent_dir = "./bc_utils/eval_videos_state_noise_v4_large"
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    save_dir = os.path.join(parent_dir, env_id.split('-')[0])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # save stats
    stats = defaultdict(list)
    action_diagnostics = []  # Track action statistics

    for ep in range(num_episodes):
        env = make_eval_env(env_id, seed=ep, render=True)
        obs, _ = env.reset()
        obs_tensor = torch.as_tensor(flatten_obs(obs), dtype=torch.float32, device=device)

        frames = []
        ep_reward = 0.0
        success = False
        out_of_bounds_count = 0

        for t in range(max_steps):

            # do forward pass on GPU
            with torch.no_grad():
                pi, tanh_mean, _, _, log_std, mean = actor(obs_tensor.to(device))
                
                # Sample action from the distribution (in [-1, 1] due to tanh transform)
                normalized_action = pi.sample().squeeze(0).cpu().numpy()
                
                # Denormalize using DATASET bounds (policy was trained on dataset-normalized actions)
                action = denormalize_action(normalized_action, dataset_low.numpy(), dataset_high.numpy())
                
                # Clip to ENV bounds before feeding to environment
                action_clipped = np.clip(action, env_low.numpy(), env_high.numpy())
                out_of_bounds = not np.allclose(action, action_clipped)
                if out_of_bounds:
                    out_of_bounds_count += 1
                
                # Track diagnostics
                with torch.no_grad():
                    std = (torch.exp(log_std) + actor.min_std).squeeze(0).cpu().numpy()
                    action_diagnostics.append({
                        'step': t,
                        'episode': ep,
                        'normalized_action': normalized_action,
                        'denormalized_action': action,
                        'action_clipped': action_clipped,
                        'out_of_bounds': out_of_bounds,
                        'action_std': std,
                        'mean_pre_tanh': mean.squeeze(0).cpu().numpy(),
                        'action_norm': np.linalg.norm(action),
                        'normalized_action_norm': np.linalg.norm(normalized_action)
                    })
            
            # Use clipped action for environment
            action = action_clipped
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            success |= info.get("success", False)

            frame = env.render()
            if frame is not None:
                frame = frame.cpu().numpy()
                frames.append(frame.squeeze(0))

            if terminated or truncated:
                break
            
            obs_tensor = torch.as_tensor(flatten_obs(obs), dtype=torch.float32, device=device)

        # save video
        video_path = f"{save_dir}/ep_{ep:03d}.mp4"
        frames = np.stack(frames, axis=0)
        imageio.mimsave(video_path, frames, fps=30)

        # Log stats
        stats["reward"].append(ep_reward)
        stats["success"].append(float(success))
        stats["length"].append(t + 1)
        stats["out_of_bounds"].append(out_of_bounds_count)

        print(
            f"EP {ep:02d} | reward {ep_reward.cpu().item():.2f} | "
            f"success {success} | steps {t+1} | out_of_bounds {out_of_bounds_count}"
        )

        env.close()

    # Print action space diagnostics after evaluation
    if action_diagnostics:
        action_diag_array = np.array([d['denormalized_action'] for d in action_diagnostics])
        action_clipped_array = np.array([d['action_clipped'] for d in action_diagnostics])
        normalized_action_array = np.array([d['normalized_action'] for d in action_diagnostics])
        
        print(f"\n=== EVALUATION ACTION STATISTICS ===")
        print(f"\nOriginal (denormalized) actions:")
        print(f"  Mean per dim: {action_diag_array.mean(axis=0)}")
        print(f"  Std per dim: {action_diag_array.std(axis=0)}")
        print(f"  Min per dim: {action_diag_array.min(axis=0)}")
        print(f"  Max per dim: {action_diag_array.max(axis=0)}")
        print(f"  Mean L2 norm: {np.mean([d['action_norm'] for d in action_diagnostics]):.3f}")
        
        print(f"\nNormalized (pre-denorm) actions:")
        print(f"  Mean per dim: {normalized_action_array.mean(axis=0)}")
        print(f"  Std per dim: {normalized_action_array.std(axis=0)}")
        print(f"  Mean L2 norm: {np.mean([d['normalized_action_norm'] for d in action_diagnostics]):.3f}")
        
        print(f"\nClipped actions:")
        print(f"  Mean per dim: {action_clipped_array.mean(axis=0)}")
        print(f"  Std per dim: {action_clipped_array.std(axis=0)}")
        
        out_of_bounds_total = sum(1 for d in action_diagnostics if d['out_of_bounds'])
        print(f"\nOut of bounds: {out_of_bounds_total} / {len(action_diagnostics)} steps ({100*out_of_bounds_total/len(action_diagnostics):.1f}%)")
        print(f"=" * 50)
    
    # Print summary stats
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Average reward: {np.mean(stats['reward']):.3f}")
    print(f"Success rate: {np.mean(stats['success']):.1%}")
    print(f"Average episode length: {np.mean(stats['length']):.1f}")
    print(f"Average out-of-bounds steps per episode: {np.mean(stats['out_of_bounds']):.1f}")
    print(f"=" * 50)
    
    # Replay expert states and measure MSE
    replay_expert_and_measure_mse(actor, dataset_low, dataset_high, env_low, env_high, device=device)

test('PushCube-v1')