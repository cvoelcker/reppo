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


def denormalize_action(normalized_action, low, high):
    """
    Denormalize actions from [-1, 1] back to original action space [low, high].
    """
    denormalized = (normalized_action + 1.0) * (high - low) / 2.0 + low
    return denormalized


def train_one_epoch(epoch_index, tb_writer, low, high, actor):
    running_loss = 0.
    last_loss = 0.
    num_train_batches = 0
    train_loss_sum = 0.0

    # NEW: accumulate diagnostics
    policy_std_sum = 0.0
    mean_action_norm_sum = 0.0
    tanh_mean_action_norm_sum = 0.0      # NEW: track post-tanh mean
    action_l2_error_sum = 0.0          # NEW: BC sanity check
    sampled_action_std_sum = 0.0       # NEW: post-tanh proxy
    policy_std_pre_tanh_sum = 0.0
    action_std_post_tanh_sum = 0.0
    expert_action_std_sum = 0.0        # NEW: track expert action variance
    denorm_expert_action_std_sum = 0.0 # NEW: track denormalized expert variance

    for i, data in enumerate(train_loader):
        obs, expert_action = data['observations'], data['actions']
        expert_action, obs = expert_action.to(device), obs.to(device)
        
        # Track original expert action statistics BEFORE normalization
        original_expert_action_std = expert_action.std(dim=0).mean().item()
        denorm_expert_action_std_sum += original_expert_action_std

        # normalize and clamp the actions to stay between -1 and +1 excluding the boundary points
        expert_action = 2.0 * (expert_action - low) / (high - low + 1e-6) - 1.0
        expert_action = torch.clamp(expert_action, -0.95, 0.95)
        
        # Track normalized expert action std
        normalized_expert_std = expert_action.std(dim=0).mean().item()
        expert_action_std_sum += normalized_expert_std

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Returns a distribution of actions between the range (-1, 1)
        dist, tanh_mean, _, _, log_std, mean = actor(obs)

        # Diagnostics (pre-tanh)
        mean_action_norm = mean.norm(dim=-1).mean() 
        mean_action_norm_sum += mean_action_norm.item()
        policy_std = (torch.exp(log_std) + actor.min_std).mean()
        policy_std_pre_tanh_sum += policy_std.item() 

        # Diagnostics (post-tanh)
        with torch.no_grad():

            # Track tanh-transformed mean (this should NOT be ~0) and std
            tanh_mean_action_norm = tanh_mean.norm(dim=-1).mean()
            tanh_mean_action_norm_sum += tanh_mean_action_norm.item()

            sampled_action = dist.rsample()
            sampled_action_std = sampled_action.std(dim=0).mean()
            action_std_post_tanh_sum += sampled_action_std.item()

            # BC sanity: are we fitting expert actions?
            action_l2_error = (sampled_action - expert_action).norm(dim=-1).mean()
            action_l2_error_sum += action_l2_error.item()

        # tensorBoard logging
        global_step = epoch_index * len(train_loader) + i
        tb_writer.add_scalar("Diagnosics/denorm_expert_action_std_sum", original_expert_action_std, global_step)
        tb_writer.add_scalar("Diagnosics/expert_action_std_sum", normalized_expert_std, global_step)
        tb_writer.add_scalar("Diagnostics/mean_action_norm_pre_tanh", mean_action_norm.item(), global_step)
        tb_writer.add_scalar("Diagnostics/policy_std_pre_tanh", policy_std.item(), global_step)
        tb_writer.add_scalar("Diagnostics/mean_action_norm_post_tanh", tanh_mean_action_norm.item(), global_step)
        tb_writer.add_scalar("Diagnostics/action_std_post_tanh", sampled_action_std.item(), global_step)
        tb_writer.add_scalar("Diagnostics/action_l2_error", action_l2_error.item(), global_step)

        # Extract log probability
        log_prob = dist.log_prob(expert_action)  # [B, action_dim]
        log_prob_sum = log_prob.sum(dim=-1)  # [B]

        # Main loss: Maximum likelihood
        mle_loss = -log_prob_sum.mean()

        # Auxiliary Loss - Enforce policy mean matches expert action means across individual dimensions
        # L2 loss between policy mean action and expert action
        action_match_loss = (tanh_mean - expert_action).pow(2).mean()

        # Combined loss: MLE + Action Matching + Std Regularization
        std_reg_coef = 0.10  # regularization on log std to prevent variance collapse
        action_match_coef = 0.50  # Weight of action matching to enforce per-dimension mean learning
        loss = mle_loss + action_match_coef * action_match_loss - std_reg_coef * log_std.mean()

        # storing average per-dimension log probs (BEFORE summing)
        train_loss_sum += -log_prob.mean(dim=0).detach().cpu()
        num_train_batches += 1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return (
        running_loss / len(train_loader),
        train_loss_sum / num_train_batches,
        policy_std_pre_tanh_sum / num_train_batches,
        mean_action_norm_sum / num_train_batches,
        tanh_mean_action_norm_sum / num_train_batches,      # NEW: return post-tanh mean
        action_std_post_tanh_sum / num_train_batches,
        action_l2_error_sum / num_train_batches,
        expert_action_std_sum / num_train_batches,
        denorm_expert_action_std_sum / num_train_batches
    )

# Load config
cfg_path = "../reppo/config/algorithm/reppo.yaml"
cfg = OmegaConf.load(cfg_path)

# Resolve references and recreate OmegaConf object (Not needed here)
# cfg = OmegaConf.to_container(cfg, resolve=True)
# cfg = OmegaConf.create(cfg)

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

# Initilaize BC specific variables
EPOCHS = 200
batch_size = 64
best_vloss = 1_000_000
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader, n_obs, n_act, low, high = load_demos_for_training("PushCube-v1", device = device, filter_success=True)
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
optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=float(cfg.optimizer.learning_rate)
    )
scaler = GradScaler()
        # enabled=cfg.platform.amp_enabled and cfg.platform.amp_dtype == torch.float16
    # )
train_losses_per_epoch = []
val_losses_per_epoch = []

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
    num_val_batches = 0
    val_loss_sum = 0.0
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    actor.train()
    avg_loss, avg_loss_per_dim, avg_policy_std, avg_mean_action_norm_pre, avg_mean_action_norm_post, avg_action_std_post_tanh_sum, avg_action_l2_error_sum, avg_expert_std, avg_denorm_expert_std = train_one_epoch(epoch_number, writer, low, high, actor)
    train_losses_per_epoch.append(avg_loss_per_dim)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling gradient flow
    actor.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            obs, expert_action = vdata['observations'], vdata['actions']
            # normalize and clamp the actions to stay between -1 and +1 excluding the boundary points
            expert_action = 2.0 * (expert_action - low) / (high - low + 1e-6) - 1.0
            expert_action = torch.clamp(expert_action, -0.95, 0.95)
            dist, _, _, _, _, _ = actor(obs.to(device))
            expert_action = expert_action.to(device)
            log_prob = dist.log_prob(expert_action)
            # storing average per-dimension log probs
            val_loss_sum += (-log_prob).mean(dim=0).detach().cpu()
            num_val_batches += 1

            # print('Validation:', expert_action, log_prob)
            log_prob = log_prob.sum(dim=-1)
            vloss = -log_prob.mean()
            running_vloss += vloss.item()

    val_losses_per_epoch.append(val_loss_sum / num_val_batches)

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if not os.path.exists('saved_models_state_noise'):
            os.makedirs('saved_models_state_noise')
        model_path = 'saved_models_state_noise/bc_model_{}_{}'.format(timestamp, epoch_number)
        torch.save(actor.state_dict(), model_path)

    epoch_number += 1

    print(
        f"[Diagnostics] Epoch {epoch_number:03d} | "
        f"policy_std_pre_tanh={avg_policy_std:.4f} | "
        f"mean_pre_tanh={avg_mean_action_norm_pre:.4f} | "
        f"mean_post_tanh={avg_mean_action_norm_post:.4f} | "
        f"action_std_post_tanh={avg_action_std_post_tanh_sum:.4f} | "
        f"action_l2_error={avg_action_l2_error_sum:.4f} | "
        f"expert_std={avg_expert_std:.4f}"
    )

# save the per-dimension loss plots
train_losses_per_epoch = torch.stack(train_losses_per_epoch)  # [num_epochs, action_dim]
val_losses_per_epoch = torch.stack(val_losses_per_epoch)      # [num_epochs, action_dim]

num_epochs, action_dim = train_losses_per_epoch.shape
if not os.path.exists('saved_models_state_noise/loss_plots'):
    os.mkdir('saved_models_state_noise/loss_plots')

# Plot per-dimension trends
for dim in range(action_dim):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses_per_epoch[:, dim], label="Train Loss", color="blue")
    plt.plot(range(1, num_epochs + 1), val_losses_per_epoch[:, dim], label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel(f"Mean NLL Loss (Action Dim {dim})")
    plt.title(f"Action Dimension {dim} Loss Trend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"saved_models_state_noise/loss_plots/loss_dim_{dim}.png")
    plt.close()

print(f"Saved {action_dim} plots in ./loss_plots/")
