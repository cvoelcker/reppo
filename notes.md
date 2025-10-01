Available maniskill demo dataset UIDs:
['AnymalC-Reach-v1', 'DrawTriangle-v1', 'LiftPegUpright-v1', 'PegInsertionSide-v1', 'PickCube-v1', 'PlugCharger-v1', 'PokeCube-v1', 'PullCube-v1', 'PullCubeTool-v1', 'PushCube-v1', 'PushT-v1', 'RollBall-v1', 'StackCube-v1', 'TwoRobotPickCube-v1', 'TwoRobotStackCube-v1']

1. dowloaded maniskill demos: 
- PickSingleYCB-v1
- RollBall-v1



First, install the required packages:
pip install h5py mani_skill



```python
from src.torchrl.demo_loader import load_demos_for_training
import torch

# Load demos for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
demo_data = load_demos_for_training(
    env_id="PickCube-v1",
    demo_dir="./demos",
    device=device,
    max_episodes=100,
    filter_success=True
)
```

### TensorDict Format

Demonstrations are converted to TensorDict with these keys:
- **observations**: `[T, obs_dim]` - State observations
- **next_observations**: `[T, obs_dim]` - Next state observations
- **actions**: `[T, action_dim]` - Actions taken
- **rewards**: `[T, 1]` - Rewards (computed from success if not available)
- **dones**: `[T, 1]` - Episode termination flags
- **truncations**: `[T, 1]` - Episode truncation flags
- **success**: `[T, 1]` - Success flags (if available)


