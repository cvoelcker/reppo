import os
import numpy as np
import hydra
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb
from datetime import datetime, time
from tensordict import TensorDict
from torch.amp import GradScaler
from src.network_utils.torch_models import Actor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from src.maniskill_utils.maniskill_dataloader_shabnam import load_demos_for_training

def train_one_epoch(cfg, epoch_index, tb_writer, data_low, data_high, actor):
    """Train one epoch and return actor loss plus diagnostic metrics."""
    # Accumulate metrics
    actor_loss_sum = 0.0
    nll_loss_sum = 0.0
    mse_loss_sum = 0.0
    num_batches = 0
    mean_sum = 0.0
    tanh_mean_sum = 0.0
    log_std_sum = 0.0

    for i, data in enumerate(train_loader):
        obs, expert_action = data['observations'], data['actions']
        expert_action, obs = expert_action.to(device), obs.to(device)
        num_batches += 1

        # Normalize expert actions to [-1, 1] using bounds with safety margin
        expert_action = 2.0 * (expert_action - data_low) / (data_high - data_low) - 1.0

        optimizer.zero_grad()

        dist, tanh_mean, _, _, log_std, mean = actor(obs)

        # Accumulate mean, tanh_mean, log_std
        mean_sum += mean.mean().item()
        tanh_mean_sum += tanh_mean.mean().item()
        log_std_sum += log_std.mean().item()

        log_prob = dist.log_prob(expert_action)  # [B, action_dim]
        
        # NLL: Maximum likelihood estimation
        nll_loss = -log_prob.sum(dim=-1).mean()
        nll_loss_sum += nll_loss.item()
        
        # MSE loss
        mse_loss = (tanh_mean - expert_action).pow(2).mean()
        mse_loss_sum += mse_loss.item()

        # Combined actor loss
        actor_loss = nll_loss + cfg.algorithm.bc_mse_loss_weight * mse_loss  # - std_reg_coef * log_std.mean()

        # Backward pass and optimization
        actor_loss.backward()
        optimizer.step()

        # Accumulate loss
        actor_loss_sum += actor_loss.item()

    # Compute averages
    avg_actor_loss = actor_loss_sum / num_batches
    avg_nll_loss = nll_loss_sum / num_batches
    avg_mse_loss = mse_loss_sum / num_batches
    avg_mean = mean_sum / num_batches
    avg_tanh_mean = tanh_mean_sum / num_batches
    avg_log_std = log_std_sum / num_batches
    
    return (
        avg_actor_loss,
        avg_nll_loss,
        avg_mse_loss,
        avg_mean,
        avg_tanh_mean,
        avg_log_std
    )

@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg: OmegaConf):
    """Main training function."""
    # Disable struct mode to allow CLI overrides
    OmegaConf.set_struct(cfg, False)
    
    # Initialize wandb
    wandb.init(
        config=dict(cfg),
        entity=cfg.logging.entity,
        project=cfg.logging.project,
        name=f"bc_pretrain_no_clamp_linear_scale_no_envstates_{cfg.env.name}",
        mode=cfg.logging.mode
    )
    
    global device, train_loader, val_loader, optimizer
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    # Initilaize BC specific variables
    EPOCHS = 128
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_vloss = 1_000_000

    # Load demo parameters from config
    env_name = cfg.env.name
    batch_size = cfg.env.demo.batch_size
    demo_path = cfg.env.demo.demo_path
    max_episodes = cfg.env.demo.max_episodes
    filter_success = cfg.env.demo.filter_success

    # Load demonstrations using config parameters
    train_loader, val_loader, n_obs, n_act, _, _ = load_demos_for_training(
        env_id=env_name,
        device=device,
        bsize=batch_size,
        demo_path=demo_path,
        max_episodes=max_episodes,
        filter_success=filter_success
    )
    # print(n_obs, n_act)
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
    ).to(device)

    temp_env = gym.make(env_name, obs_mode="state_dict", control_mode=cfg.env.get('control_mode'))
    low = torch.from_numpy(temp_env.action_space.low).float().to(device)
    high = torch.from_numpy(temp_env.action_space.high).float().to(device)
    temp_env.close()
    print(f"True environment bounds: low={low.cpu().numpy()}, high={high.cpu().numpy()}")
    
    optimizer = optim.AdamW(
            list(actor.parameters()),
            lr=float(cfg.algorithm.optimizer.learning_rate)
        )
    scaler = GradScaler()
    train_losses = []
    val_losses = []

    # Check dataset action stats
    all_actions = torch.cat(
        [batch["actions"] for batch in train_loader] +
        [batch['actions'] for batch in val_loader],
        dim=0
    )
    
    # Compute empirical bounds from dataset
    data_low = all_actions.min(dim=0).values
    data_high = all_actions.max(dim=0).values

    # Add 10% safety margin to bounds
    margin = 0.1 * (data_high - data_low)
    low_with_margin = data_low - margin
    high_with_margin = data_high + margin
    
    # Ensure bounds are at least [-1, 1] in each dimension
    low = torch.min(low_with_margin, torch.full_like(low_with_margin, -1.0))
    high = torch.max(high_with_margin, torch.full_like(high_with_margin, 1.0))

    # Normalize actions using bounds with safety margin
    normalized_actions = 2.0 * (all_actions - low) / (high - low) - 1.0
    
    expert_action_min = all_actions.min().item()
    expert_action_max = all_actions.max().item()
    normalized_action_min = normalized_actions.min().item()
    normalized_action_max = normalized_actions.max().item()
    
    print("\n=== DATASET ACTION STATS ===")
    print(f"Expert action min (raw): {expert_action_min:.6f}")
    print(f"Expert action max (raw): {expert_action_max:.6f}")
    print(f"Normalized action min: {normalized_action_min:.6f}")
    print(f"Normalized action max: {normalized_action_max:.6f}")
    print(f"Action normalization bounds (low): {low.cpu().numpy()}")
    print(f"Action normalization bounds (high): {high.cpu().numpy()}")
    print("================================\n")
    time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch_number + 1}/{EPOCHS}')

        # training
        actor.train()
        avg_actor_loss, avg_nll_loss, avg_mse_loss, avg_mean, avg_tanh_mean, avg_log_std = train_one_epoch(cfg, epoch_number, writer, low, high, actor)
        train_losses.append(avg_actor_loss)

        # validation
        actor.eval()
        val_loss_sum = 0.0
        val_num_batches = 0
        val_nll_loss = 0.0
        val_mse_loss = 0.0
        val_mean = 0.0
        val_tanh_mean = 0.0
        val_log_std = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                obs, expert_action = vdata['observations'], vdata['actions']
                # Normalize expert actions to [-1, 1] using bounds with safety margin
                expert_action = 2.0 * (expert_action - low) / (high - low) - 1.0
                
                obs = obs.to(device)
                expert_action = expert_action.to(device)
                val_num_batches += 1
                
                dist, tanh_mean, _, _, log_std, mean = actor(obs)
                log_prob = dist.log_prob(expert_action)
                
                # NLL: Maximum likelihood estimation
                nll_loss = -log_prob.sum(dim=-1).mean()
                val_nll_loss += nll_loss.item()
                
                # MSE loss
                mse_loss = (tanh_mean - expert_action).pow(2).mean()
                val_mse_loss += mse_loss.item()
                
                val_mean += mean.mean().item()
                val_tanh_mean += tanh_mean.mean().item()
                val_log_std += log_std.mean().item()

                # Combined actor loss
                val_loss = nll_loss + cfg.algorithm.bc_mse_loss_weight * mse_loss
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / val_num_batches
        val_losses.append(avg_val_loss)
        
        val_nll_loss /= val_num_batches
        val_mse_loss /= val_num_batches
        val_mean /= val_num_batches
        val_tanh_mean /= val_num_batches
        val_log_std /= val_num_batches

        print(f'Loss: train={avg_actor_loss:.4f} | val={avg_val_loss:.4f}')

        # Log to wandb
        wandb.log({
            "train/nll_loss": avg_nll_loss,
            "train/mse_loss": avg_mse_loss,
            "train/mean": avg_mean,
            "train/tanh_mean": avg_tanh_mean,
            "train/log_std": avg_log_std,
            "val/nll_loss": val_nll_loss,
            "val/mse_loss": val_mse_loss,
            "val/mean": val_mean,
            "val/tanh_mean": val_tanh_mean,
            "val/log_std": val_log_std,
            "epoch": epoch_number + 1
        }, step=epoch_number + 1)

        # Save best actor based on validation actor loss
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_dir = f'{env_name}/saved_models/{timestamp}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/bc_model_actor_best.pt'
            torch.save(actor.state_dict(), model_path)
            print(f'  ✓ Saved best actor model to {model_path} (val_loss={avg_val_loss:.4f})')

        epoch_number += 1

    print(f"\n{'='*60}")
    print(f"Training Complete! Best validation loss: {best_vloss:.4f}")
    print(f"{'='*60}")

    # Save final loss plots
    plot_dir = f'{env_name}/loss_plots/{timestamp}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot actor losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f"Train Loss for {env_name}", color="blue")
    plt.plot(val_losses, label=f"Val Loss for {env_name}", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Actor Loss")
    plt.legend()
    plt.title(f'Losses for {env_name}')
    plot_path = f'{plot_dir}/loss_plot.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved loss plots to {plot_path}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()