import os
import numpy as np
import hydra
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from tensordict import TensorDict
from torch.amp import GradScaler
from src.network_utils.torch_models import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from src.maniskill_utils.maniskill_dataloader_shabnam import load_demos_for_training

def denormalize_action(normalized_action, low, high):
    """
    Denormalize actions from [-1, 1] back to original action space [low, high].
    """
    denormalized = (normalized_action + 1.0) * (high - low) / 2.0 + low
    return denormalized

def train_one_epoch(epoch_index, tb_writer, low, high, actor, critic):
    """Train one epoch and return average actor and critic losses."""
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        obs, expert_action, reward = data['observations'], data['actions'], data['rewards']
        next_obs = data['next_observations']  # Get next_obs from dataset, not from next batch
        
        expert_action, obs, reward = expert_action.to(device), obs.to(device), reward.to(device)
        next_obs = next_obs.to(device)
        num_batches += 1

        # Normalize actions to [-1, 1]
        expert_action = 2.0 * (expert_action - low) / (high - low + 1e-6) - 1.0
        expert_action = torch.clamp(expert_action, -0.95, 0.95)

        # Zero gradients
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        # ===== ACTOR LOSS =====
        dist, tanh_mean, _, _, _, _ = actor(obs)
        log_prob = dist.log_prob(expert_action)  # [B, action_dim]
        
        # NLL: Maximum likelihood estimation
        nll_loss = -log_prob.sum(dim=-1).mean()
        
        # MSE: Explicit mean matching
        mse_loss = (tanh_mean - expert_action).pow(2).mean()
        
        # Combined actor loss
        actor_loss = nll_loss + mse_loss

        # ===== CRITIC LOSS =====
        # Get Q-values under offline policy (behavioral policy from data)
        value, _, _, _ = critic(obs, expert_action) # Qpi(s, a)
        
        # Use actual next observations from dataset, not next batch
        with torch.no_grad():
            next_dist, _, _, _, _, _ = actor(next_obs)
            next_actions = next_dist.rsample()  # Sample next actions from policy
            next_value, _, _, _ = critic(next_obs, next_actions) # Qpi(s', a')

        # TD error = r + γ * Q(s', a') - Q(s, a)
        reward = reward.unsqueeze(-1)  # Shape [B] -> [B, 1] for broadcasting
        td_error = reward + cfg.gamma * next_value - value
        td_loss = 0.5 * td_error.pow(2).mean()

        # CQL regularization: penalize overestimation on the current state
        online_actions = dist.rsample()  # Sample from current policy on current state
        online_q_values, _, _, _ = critic(obs, online_actions) # Qmu(s, a)
        # When E[Qpi(s, a)] - E[Qmu(s, a)] is large and positive: The critic is overestimating the policy's actions compared to offline data → ADD penalty to loss → increase loss → reduce Q-values
        cql_penalty = (online_q_values.mean() - value.mean()).detach()  
        # Combined critic loss: TD loss + CQL penalty
        critic_loss = td_loss + 2 * cql_penalty

        # Backward pass and optimization
        actor_loss.backward()
        critic_loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()

        # Accumulate losses
        actor_loss_sum += actor_loss.item()
        critic_loss_sum += critic_loss.item()

    avg_actor_loss = actor_loss_sum / num_batches
    avg_critic_loss = critic_loss_sum / num_batches
    
    return avg_actor_loss, avg_critic_loss

# Load config
cfg_path = "../reppo/config/algorithm/reppo.yaml"
cfg = OmegaConf.load(cfg_path)

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

# Initilaize BC specific variables
EPOCHS = 200
batch_size = 64
best_actor_vloss, best_critic_vloss = 1_000_000, 1_000_000
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

critic = Critic(
        n_obs=n_obs,
        n_act=n_act,
        num_atoms=10,
        vmin=-15.0,
        vmax=15.0,
        hidden_dim=cfg.critic_hidden_dim,
        encoder_layers=cfg.num_critic_encoder_layers,
        head_layers=cfg.num_critic_head_layers,
        pred_layers=cfg.num_critic_pred_layers,
        device=device,
    ).to(device)

# Learnable dimension weights for weighted loss
# These will be learned during training to weight important dimensions (e.g., gripper)
# Used in both NLL and MSE losses
# dim_weights_param = torch.nn.Parameter(torch.ones(n_act, device=device))
# actor.register_parameter('dim_weights', dim_weights_param)

optimizer_actor = optim.AdamW(
        list(actor.parameters()),
        lr=float(cfg.optimizer.learning_rate)
    )
optimizer_critic = optim.AdamW(
        list(critic.parameters()),
        lr=float(cfg.optimizer.learning_rate)
    )
scaler = GradScaler()
train_actor_losses = []
train_critic_losses = []
val_actor_losses = []
val_critic_losses = []

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
    critic.train()
    avg_actor_loss, avg_critic_loss = train_one_epoch(epoch_number, writer, low, high, actor, critic)
    train_actor_losses.append(avg_actor_loss)
    train_critic_losses.append(avg_critic_loss)

    # ===== VALIDATION PHASE =====
    actor.eval()
    critic.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        val_actor_loss_sum = 0.0
        val_critic_loss_sum = 0.0
        val_num_batches = 0
        
        for i, vdata in enumerate(val_loader):
            obs, expert_action, reward = vdata['observations'], vdata['actions'], vdata['rewards']
            next_obs = vdata['next_observations']  #   Get next_obs from dataset
            
            # Normalize actions to [-1, 1]
            expert_action = 2.0 * (expert_action - low) / (high - low + 1e-6) - 1.0
            expert_action = torch.clamp(expert_action, -0.95, 0.95)
            reward = reward.unsqueeze(-1)  # Shape [B] -> [B, 1] for broadcasting
            obs = obs.to(device)
            expert_action = expert_action.to(device)
            reward = reward.to(device)
            next_obs = next_obs.to(device)
            val_num_batches += 1
            
            dist, tanh_mean, _, _, _, _ = actor(obs)
            
            # ===== ACTOR LOSS =====
            log_prob = dist.log_prob(expert_action)
            nll_loss = -log_prob.sum(dim=-1).mean()
            mse_loss = (tanh_mean - expert_action).pow(2).mean()
            val_actor_loss = nll_loss + mse_loss
            val_actor_loss_sum += val_actor_loss.item()
            
            # ===== CRITIC LOSS =====
            value, _, _, _ = critic(obs, expert_action)
            
            #  Use actual next observations from dataset
            with torch.no_grad():
                next_dist, _, _, _, _, _ = actor(next_obs)
                next_actions = next_dist.rsample()
                next_value, _, _, _ = critic(next_obs, next_actions)
            
            reward_unsqueezed = reward.unsqueeze(-1) if reward.dim() == 1 else reward  # Ensure [B, 1]
            td_error = reward_unsqueezed + cfg.gamma * next_value - value
            td_loss = 0.5 * td_error.pow(2).mean()
            
            online_actions = dist.rsample()
            online_q_values, _, _, _ = critic(obs, online_actions)
            cql_penalty = (online_q_values.mean() - value.mean()).detach()
            
            val_critic_loss = td_loss + 2 * cql_penalty
            val_critic_loss_sum += val_critic_loss.item()

    avg_val_actor_loss = val_actor_loss_sum / val_num_batches
    avg_val_critic_loss = val_critic_loss_sum / val_num_batches
    val_actor_losses.append(avg_val_actor_loss)
    val_critic_losses.append(avg_val_critic_loss)

    # Print epoch results
    print(f'  Actor  Loss: train={avg_actor_loss:.4f} | val={avg_val_actor_loss:.4f}')
    print(f'  Critic Loss: train={avg_critic_loss:.4f} | val={avg_val_critic_loss:.4f}')

    # TensorBoard logging
    writer.add_scalars('Actor Loss', {'train': avg_actor_loss, 'val': avg_val_actor_loss}, epoch_number + 1)
    writer.add_scalars('Critic Loss', {'train': avg_critic_loss, 'val': avg_val_critic_loss}, epoch_number + 1)
    writer.flush()

    # Save best actor based on validation actor loss
    if avg_val_actor_loss < best_actor_vloss:
        best_actor_vloss = avg_val_actor_loss
        if not os.path.exists('../saved_models_AC_alpha_2'):
            os.makedirs('../saved_models_AC_alpha_2')
        model_path = f'../saved_models_AC_alpha_2/bc_model_actor_{timestamp}_{epoch_number}'
        torch.save(actor.state_dict(), model_path)
        print(f'  ✓ Saved best actor model')

    if avg_val_critic_loss < best_critic_vloss:
        best_critic_vloss = avg_val_critic_loss
        if not os.path.exists('../saved_models_AC_alpha_2'):
            os.makedirs('../saved_models_AC_alpha_2')
        model_path = f'../saved_models_AC_alpha_2/bc_model_critic_{timestamp}_{epoch_number}'
        torch.save(critic.state_dict(), model_path)
        print(f'  ✓ Saved best critic model')

    epoch_number += 1

# ===== TRAINING COMPLETE =====
print(f"\n{'='*60}")
print(f"Training Complete! Best validation actor loss: {best_actor_vloss:.4f}")
print(f"Training Complete! Best validation critic loss: {best_critic_vloss:.4f}")
print(f"{'='*60}")

# Save final loss plots
if not os.path.exists('../saved_models_AC_alpha_2/loss_plots'):
    os.makedirs('../saved_models_AC_alpha_2/loss_plots')

# Plot actor losses
plt.figure(figsize=(10, 5))
plt.plot(train_actor_losses, label="Train Actor Loss", color="blue")
plt.plot(val_actor_losses, label="Val Actor Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Actor Loss")
plt.title("Actor Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../saved_models_AC_alpha_2/loss_plots/actor_loss.png")
plt.close()

# Plot critic losses
plt.figure(figsize=(10, 5))
plt.plot(train_critic_losses, label="Train Critic Loss", color="blue")
plt.plot(val_critic_losses, label="Val Critic Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Critic Loss")
plt.title("Critic Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../saved_models_AC_alpha_2/loss_plots/critic_loss.png")
plt.close()

print(f"Saved loss plots to ../saved_models_AC_alpha_2/loss_plots/")