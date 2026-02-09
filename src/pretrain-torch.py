import os
import numpy as np
import hydra
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from tensordict import TensorDict
from torch.amp import GradScaler
from src.network_utils.torch_models import Actor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from src.maniskill_utils.maniskill_dataloader_shabnam import load_demos_for_training

def train_one_epoch(epoch_index, tb_writer, low, high, actor):
    """Train one epoch and return actor loss plus diagnostic metrics."""
    actor_loss_sum = 0.0
    num_batches = 0

    # Accumulate diagnostics
    policy_std_sum = 0.0
    mean_action_norm_sum = 0.0
    tanh_mean_action_norm_sum = 0.0
    action_l2_error_sum = 0.0
    action_std_post_tanh_sum = 0.0
    expert_action_std_sum = 0.0
    denorm_expert_action_std_sum = 0.0

    for i, data in enumerate(train_loader):
        obs, expert_action = data['observations'], data['actions']
        expert_action, obs = expert_action.to(device), obs.to(device)
        num_batches += 1
        
        # Normalize raw actions from environment bounds to [-1, 1]
        expert_action = 2.0 * (expert_action - low) / (high - low) - 1.0
        
        # Track original expert action statistics (before clamping)
        original_expert_action_std = expert_action.std(dim=0).mean().item()
        denorm_expert_action_std_sum += original_expert_action_std

        # Clamp normalized actions for numerical stability
        expert_action = torch.clamp(expert_action, -0.95, 0.95)
        
        # Track normalized expert action std
        normalized_expert_std = expert_action.std(dim=0).mean().item()
        expert_action_std_sum += normalized_expert_std

        # Zero gradients
        optimizer.zero_grad()

        # ===== ACTOR LOSS =====
        dist, tanh_mean, _, _, log_std, mean = actor(obs)

        # Diagnostics (pre-tanh)
        mean_action_norm = mean.norm(dim=-1).mean() 
        mean_action_norm_sum += mean_action_norm.item()
        policy_std = (torch.exp(log_std) + actor.min_std).mean()
        policy_std_sum += policy_std.item() 

        # Diagnostics (post-tanh)
        with torch.no_grad():
            tanh_mean_action_norm = tanh_mean.norm(dim=-1).mean()
            tanh_mean_action_norm_sum += tanh_mean_action_norm.item()

            sampled_action = dist.rsample()
            sampled_action_std = sampled_action.std(dim=0).mean()
            action_std_post_tanh_sum += sampled_action_std.item()

            # BC sanity: are we fitting expert actions?
            action_l2_error = (sampled_action - expert_action).norm(dim=-1).mean()
            action_l2_error_sum += action_l2_error.item()

        # TensorBoard logging (per-batch diagnostics)
        global_step = epoch_index * len(train_loader) + i
        tb_writer.add_scalar("Diagnostics/denorm_expert_action_std", original_expert_action_std, global_step)
        tb_writer.add_scalar("Diagnostics/expert_action_std", normalized_expert_std, global_step)
        tb_writer.add_scalar("Diagnostics/mean_action_norm_pre_tanh", mean_action_norm.item(), global_step)
        tb_writer.add_scalar("Diagnostics/policy_std_pre_tanh", policy_std.item(), global_step)
        tb_writer.add_scalar("Diagnostics/mean_action_norm_post_tanh", tanh_mean_action_norm.item(), global_step)
        tb_writer.add_scalar("Diagnostics/action_std_post_tanh", sampled_action_std.item(), global_step)
        tb_writer.add_scalar("Diagnostics/action_l2_error", action_l2_error.item(), global_step)

        # Compute loss
        log_prob = dist.log_prob(expert_action)  # [B, action_dim]
        
        # NLL: Maximum likelihood estimation
        nll_loss = -log_prob.sum(dim=-1).mean()
        
        # MSE: Explicit mean matching
        mse_loss = (tanh_mean - expert_action).pow(2).mean()

        # std_reg_coef = 0.10  # regularization on log std to prevent variance collapse
        action_match_coef = 1 # Weight of action matching to enforce per-dimension mean learning
        
        # Combined actor loss
        actor_loss = nll_loss + action_match_coef * mse_loss  # - std_reg_coef * log_std.mean()

        # Backward pass and optimization
        actor_loss.backward()
        optimizer.step()

        # Accumulate loss
        actor_loss_sum += actor_loss.item()

    # Compute averages
    avg_actor_loss = actor_loss_sum / num_batches
    avg_policy_std = policy_std_sum / num_batches
    avg_mean_action_norm_pre = mean_action_norm_sum / num_batches
    avg_mean_action_norm_post = tanh_mean_action_norm_sum / num_batches
    avg_action_std_post_tanh = action_std_post_tanh_sum / num_batches
    avg_action_l2_error = action_l2_error_sum / num_batches
    avg_expert_std = expert_action_std_sum / num_batches
    avg_denorm_expert_std = denorm_expert_action_std_sum / num_batches
    
    return (
        avg_actor_loss,
        avg_policy_std,
        avg_mean_action_norm_pre,
        avg_mean_action_norm_post,
        avg_action_std_post_tanh,
        avg_action_l2_error,
        avg_expert_std,
        avg_denorm_expert_std
    )

# Load config
cfg_path = "../reppo/config/algorithm/reppo.yaml"
cfg = OmegaConf.load(cfg_path)

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

# Initilaize BC specific variables
env_id = 'PickCube-v1'
control_mode = "pd_joint_delta_pos"
demo_path = '/scratch/cluster/idutta/h5_files/PickCube/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5'
EPOCHS = 200
batch_size = 64
best_vloss = 1_000_000
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader, n_obs, n_act = load_demos_for_training(env_id, demo_path = demo_path, device = device, filter_success=True)

# Get TRUE environment bounds, not dataset empirical bounds. The expert actions would be normalized from this space to [-1, 1] space.
import gymnasium as gym
temp_env = gym.make(env_id, obs_mode="state_dict", control_mode=control_mode)
low = torch.from_numpy(temp_env.action_space.low).float().to(device)
high = torch.from_numpy(temp_env.action_space.high).float().to(device)
temp_env.close()

print(f"Using environment bounds for normalization: low={low.cpu().numpy()}, high={high.cpu().numpy()}")
# print(n_obs, n_act)
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
).to(device)

# Learnable dimension weights for weighted loss
# These will be learned during training to weight important dimensions (e.g., gripper)
# Used in both NLL and MSE losses
# dim_weights_param = torch.nn.Parameter(torch.ones(n_act, device=device))
# actor.register_parameter('dim_weights', dim_weights_param)

optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=float(cfg.optimizer.learning_rate)
    )
scaler = GradScaler()
train_losses = []
val_losses = []

# Check dataset action stats
all_actions = torch.cat(
    [batch["actions"] for batch in train_loader],
    dim=0
)

print("\n=== DATASET ACTION STATS ===")
print("Action mean per dim:", all_actions.mean(dim=0))
print("Action std  per dim:", all_actions.std(dim=0))
print("Mean action L2 norm:", all_actions.norm(dim=-1).mean().item())
print("================================\n")

for epoch in range(EPOCHS):
    print(f'\nEPOCH {epoch_number + 1}/{EPOCHS}')

    # ===== TRAINING PHASE =====
    actor.train()
    avg_actor_loss, avg_policy_std, avg_mean_action_norm_pre, avg_mean_action_norm_post, avg_action_std_post_tanh, avg_action_l2_error, avg_expert_std, avg_denorm_expert_std = train_one_epoch(epoch_number, writer, low, high, actor)
    train_losses.append(avg_actor_loss)

    # ===== VALIDATION PHASE =====
    actor.eval()
    val_loss_sum = 0.0
    val_num_batches = 0

    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            obs, expert_action = vdata['observations'], vdata['actions']
            # Normalize raw actions from environment bounds to [-1, 1]
            expert_action = 2.0 * (expert_action - low) / (high - low) - 1.0
            # Clamp normalized actions for numerical stability
            expert_action = torch.clamp(expert_action, -0.95, 0.95)
            obs = obs.to(device)
            expert_action = expert_action.to(device)
            val_num_batches += 1
            
            # ===== ACTOR LOSS =====
            dist, tanh_mean, _, _, log_std, _ = actor(obs)
            log_prob = dist.log_prob(expert_action)
            
            # NLL: Maximum likelihood estimation
            nll_loss = -log_prob.sum(dim=-1).mean()
            
            # MSE: Explicit mean matching
            mse_loss = (tanh_mean - expert_action).pow(2).mean()

            # std_reg_coef = 0.10  # regularization on log std to prevent variance collapse
            action_match_coef = 1 # Weight of action matching to enforce per-dimension mean learning
            
            # Combined actor loss
            val_loss = nll_loss + action_match_coef * mse_loss  # - std_reg_coef * log_std.mean()
            val_loss_sum += val_loss.item()

    avg_val_loss = val_loss_sum / val_num_batches
    val_losses.append(avg_val_loss)

    # Print epoch results
    print(f'  Loss: train={avg_actor_loss:.4f} | val={avg_val_loss:.4f}')

    # TensorBoard logging
    writer.add_scalars('Actor Loss', {'train': avg_actor_loss, 'val': avg_val_loss}, epoch_number + 1)
    writer.flush()

    # Save best actor based on validation actor loss
    if avg_val_loss < best_vloss:
        best_vloss = avg_val_loss
        if not os.path.exists(f'../saved_models_state_goal/{env_id}'):
            os.makedirs(f'../saved_models_state_goal/{env_id}')
        model_path = f'../saved_models_state_goal/{env_id}/bc_model_actor_{timestamp}_{epoch_number}'
        torch.save(actor.state_dict(), model_path)
        print(f'  ✓ Saved best actor model')

    epoch_number += 1

    # Print diagnostics
    print(
        f"  [Diagnostics] "
        f"policy_std={avg_policy_std:.4f} | "
        f"mean_pre_tanh={avg_mean_action_norm_pre:.4f} | "
        f"mean_post_tanh={avg_mean_action_norm_post:.4f} | "
        f"action_std_post_tanh={avg_action_std_post_tanh:.4f} | "
        f"action_l2_error={avg_action_l2_error:.4f} | "
        f"expert_std={avg_expert_std:.4f}"
    )

# ===== TRAINING COMPLETE =====
print(f"\n{'='*60}")
print(f"Training Complete! Best validation loss: {best_vloss:.4f}")
print(f"{'='*60}")

# Save final loss plots
if not os.path.exists(f'../saved_models_state_goal/{env_id}/loss_plots'):
    os.makedirs(f'../saved_models_state_goal/{env_id}/loss_plots')

# Plot actor losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Val Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Actor Loss")
plt.title("Actor Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../saved_models_state_goal/{env_id}/loss_plots/actor_loss.png")
plt.close()

print(f"Saved loss plots to ../saved_models_state_goal/{env_id}/loss_plots/")