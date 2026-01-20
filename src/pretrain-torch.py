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

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    num_train_batches = 0
    train_loss_sum = 0.0

    # Here, we use enumerate(train_loader) instead of iter(train_loader) so that we can track the batch index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # print(data)
        
        obs, expert_action = data['observations'], data['actions']
        expert_action, obs = expert_action.to(device), obs.to(device)
        # normalize and clamp the actions to stay between -1 and +1 excluding the boundary points
        expert_action = 2.0 * (expert_action - low) / (high - low + 1e-6) - 1.0
        expert_action = torch.clamp(expert_action, -0.95, 0.95)
        # print(expert_action.shape, obs.shape)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):

        # Returns a distribution of actions between the range (-1, 1). It transforms the bell-curve shape of the normal distribution into a different, S-shaped curve, making it more concentrated near zero and rapidly flattening towards -1 and 1
        dist, _, _, _ = actor(obs)

        # Extract the log probability of the actions instead of raw probabilities (for more stability and better gradient flow especially while dealing with very small probailities)
        # print(expert_action)
        log_prob = dist.log_prob(expert_action)  # [B, action_dim]

        # storing average per-dimension log probs
        train_loss_sum += -log_prob.mean(dim=0).detach().cpu()
        num_train_batches += 1

        log_prob = log_prob.sum(dim=-1)           # [B]

        # Maximize the negative log likelihood as the loss. LBC​=−E[logπθ​(a∣s)]​
        loss = -log_prob.mean()
        # print('Training:', expert_action, log_prob)

        # Scale loss before backward()
        # scaler.scale(loss).backward()
        loss.backward()

        # Step the optimizer using the scaler
        # scaler.step(optimizer)
        optimizer.step()

        # Update the scaler's scale factor
        # scaler.update()

        # Gather data and report
        running_loss += loss.item()
        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return running_loss / len(train_loader), train_loss_sum / num_train_batches

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

for epoch in range(EPOCHS):
    num_val_batches = 0
    val_loss_sum = 0.0
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    actor.train()
    avg_loss, avg_loss_per_dim = train_one_epoch(epoch_number, writer)
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
            dist, _, _, _ = actor(obs.to(device))
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
        if not os.path.exists('saved_models_v3'):
            os.makedirs('saved_models_v3')
        model_path = 'saved_models_v3/bc_model_{}_{}'.format(timestamp, epoch_number)
        torch.save(actor.state_dict(), model_path)

    epoch_number += 1

# save the per-dimension loss plots
train_losses_per_epoch = torch.stack(train_losses_per_epoch)  # [num_epochs, action_dim]
val_losses_per_epoch = torch.stack(val_losses_per_epoch)      # [num_epochs, action_dim]

num_epochs, action_dim = train_losses_per_epoch.shape
if not os.path.exists('saved_models_v3/loss_plots'):
    os.mkdir('saved_models_v3/loss_plots')

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
    plt.savefig(f"saved_models_v3/loss_plots/loss_dim_{dim}.png")
    plt.close()

print(f"Saved {action_dim} plots in ./loss_plots/")