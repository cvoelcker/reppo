import torch
import torch.nn as nn
import torch.optim as optim
import mani_skill.envs
import gymnasium as gym
from src.torchrl.demo_loader import DemoConfig, load_demos_for_training

# Load the demos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
demo_td = load_demos_for_training(env_id="RollBall-v1", demo_dir="/home/tajiksh1/.maniskill/demos", device=device)

# observations and actions
obs = demo_td["observations"].to(device)   # [T, obs_dim]
acts = demo_td["actions"].to(device)       # [T, act_dim]

print("Obs shape:", obs.shape)
print("Acts shape:", acts.shape)
print("Obs sample:", obs[:1])
print("Acts sample:", acts[:1])

#  simple MLP policy
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self, x):
        return self.net(x)

policy = Policy(obs.shape[1], acts.shape[1]).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
criterion = nn.MSELoss()   # assuming continuous actions

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    pred_actions = policy(obs)
    loss = criterion(pred_actions, acts)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save policy
torch.save(policy.state_dict(), "mani_skill_policy.pth")
