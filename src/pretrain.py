import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
import torch
from omegaconf import OmegaConf
import hydra
from flax import nnx
from flax import serialization as flax_serialization

from src.torchrl.demo_loader import load_demos_for_training
from gymnax.environments.spaces import Box
from src.algorithms.reppo.ff_reppo import make_init_fn
from src.normalization import Normalizer

@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Load demo dataset
    demo_dir = cfg.env.demonstrations.demo_dir
    env_id = cfg.env.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = load_demos_for_training(env_id=env_id, demo_dir=demo_dir, device=device, filter_success=True)

    obs = td["observations"]  # [N, obs_dim]
    acts = td["actions"]      # [N, act_dim]
    obs_dim = obs.shape[1]
    act_dim = acts.shape[1]

    print(f"Demo data: {obs.shape[0]} transitions, obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    # Convert to numpy for jax
    obs_np = obs.detach().cpu().numpy()
    acts_np = acts.detach().cpu().numpy()
    # Compute per-dimension action scale so we can train with actions in [-1,1]
    action_max_abs = np.max(np.abs(acts_np), axis=0)
    # Keep scale >= 1.0 so we don't blow up small actions.
    action_scale = np.maximum(action_max_abs, 1.0)
    acts_scaled_np = acts_np / action_scale
    # Normalize observations since reppo does
    normalizer = Normalizer()
    norm_state = normalizer.init(jnp.zeros(obs_dim))
    norm_state = normalizer.update(norm_state, jnp.array(obs_np))
    obs_norm = np.array(normalizer.normalize(norm_state, jnp.array(obs_np)))
    # Move to cuda
    obs_jax = jax.device_put(jnp.array(obs_norm))
    obs_jax = jax.device_put(jnp.array(obs_np))
    acts_jax = jax.device_put(jnp.array(acts_scaled_np))
    # Initialize policy
    key = jax.random.PRNGKey(cfg.seed)
    obs_low = jnp.full((obs_dim,), -1.0)
    obs_high = jnp.full((obs_dim,), 1.0)
    act_low = jnp.full((act_dim,), -1.0)
    act_high = jnp.full((act_dim,), 1.0)
    # box space for init 
    observation_space = Box(low=obs_low, high=obs_high, shape=(obs_dim,))
    action_space = Box(low=act_low, high=act_high, shape=(act_dim,))
    # From ff_reppo
    train_state = make_init_fn(cfg, observation_space, action_space)(key)
    actor_graphdef = train_state.actor.graphdef
    params = train_state.actor.params
    # Define apply_fn for policy
    def apply_fn(params, obs):
        model = nnx.merge(actor_graphdef, params)
        # Determine action using the model
        return model.det_action(obs)

    # MSE loss and optimizer
    lr = cfg.algorithm.optimizer.learning_rate
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Single update step
    @jax.jit
    def update_step(params, opt_state, obs_batch, acts_batch):
        # A pytree with the same structure as params, containing d(loss)/d(param)
        grads = jax.grad(lambda p: jnp.mean((apply_fn(p, obs_batch) - acts_batch)**2))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss = jnp.mean((apply_fn(params, obs_batch) - acts_batch)**2)
        return params, opt_state, loss

    # Pretraining loop
    batch_size = 256
    num_epochs = 100
    num_transitions = obs_jax.shape[0]

    for epoch in range(num_epochs):
        perm = np.random.permutation(num_transitions)
        obs_shuffled = obs_jax[perm]
        acts_shuffled = acts_jax[perm]

        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, num_transitions, batch_size):
            obs_batch = obs_shuffled[i:i+batch_size]
            acts_batch = acts_shuffled[i:i+batch_size]
            params, opt_state, batch_loss = update_step(params, opt_state, obs_batch, acts_batch)
            epoch_loss += batch_loss
            num_batches += 1

        epoch_loss /= num_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:8f}", flush=True)

    # Save pretrained policy and preprocessing metadata
    save_dir = "./pretrained"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{env_id}_pretrained_policy.npz")
    # Serialize normalization state (may be jax arrays)
    try:
        norm_mean = np.array(norm_state.mean)
        norm_var = np.array(norm_state.var)
        norm_count = np.array(norm_state.count)
    except Exception:
        norm_mean = None
        norm_var = None
        norm_count = None

    # dicts/arrays suitable for serialization
    params_state = flax_serialization.to_state_dict(params)
    # Save params as a pickled state dict (preserves nested structure reliably)
    base, _ = os.path.splitext(save_path)
    params_pkl_path = base + ".params.pkl"
    with open(params_pkl_path, "wb") as fh:
        pickle.dump(params_state, fh)
    np.savez(save_path, norm_mean=norm_mean, norm_var=norm_var, norm_count=norm_count, action_scale=action_scale)
    print(f"Pretrained policy saved to {save_path}", flush=True)
    print(f"Params pickled to {params_pkl_path}", flush=True)

if __name__ == "__main__":
    main()
