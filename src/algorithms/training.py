import logging
import gymnasium
from gymnax.environments.environment import Environment
import jax
import torch
import numpy as np
import os
from src.common import (
    EvalFn,
    InitFn,
    Key,
    LearnerFn,
    LogCallback,
    PolicyFn,
    RolloutFn,
    TrainFn,
    TrainState,
)
from src.algorithms import utils
import jax.numpy as jnp
from flax.serialization import to_state_dict
# from src.maniskill_utils.maniskill_env import OfflineDatasetEnv

from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
from src.runners.maniskill_runner import flatten_obs, get_demo_obs_keys

def make_scan_train_fn(
    # env : OfflineDatasetEnv(),
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    num_seeds: int,
    max_episode_steps: int,
    stochastic_eval: bool,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    eval_fn: EvalFn | None = None,
    rollout_fn: RolloutFn | None = None,
    log_callback: LogCallback | None = None,
    demo_path: str | None = None,
    bc_indicator: bool = False,
) -> TrainFn:
    from src.runners.maniskill_runner import (
        make_eval_fn as make_eval_fn,
        make_rollout_fn as make_rollout_fn,
    )

    # Initialize the environment and wrap it to admit vectorized behavior.
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    eval_interval = int((total_time_steps / (num_steps * num_envs)) // num_eval)

    if eval_fn is None:
        eval_fn = make_eval_fn(eval_env, max_episode_steps, demo_path=demo_path, bc_indicator=bc_indicator)

    if rollout_fn is None:
        rollout_fn = make_rollout_fn(env, num_steps=num_steps, num_envs=num_envs, demo_path=demo_path, bc_indicator=bc_indicator)

    if log_callback is None:
        log_callback = lambda state, metrics: None

    def train_step(
        state: TrainState, key: Key
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        key, rollout_key, learn_key = jax.random.split(key, 3)
        # Collect trajectories from `state`
        policy = policy_fn(state, False)
        transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )
        # Execute an update to the policy with `transitions`
        state, update_metrics = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        metrics = {**update_metrics, **update_metrics}
        state = state.replace(iteration=state.iteration + 1)
        return state, metrics

    def train_eval_step(key, train_state):
        train_key, eval_key = jax.random.split(key)
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        policy = policy_fn(train_state, not stochastic_eval)
        eval_metrics = eval_fn(eval_key, policy)
        metrics = {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

        return train_state, metrics

    def train_eval_loop_body(
        train_state: TrainState, key: Key
    ) -> tuple[TrainState, dict]:
        # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(train_eval_step)(
            jax.random.split(subkey, num_seeds), train_state
        )
        jax.debug.callback(log_callback, train_state, metrics)
        return train_state, metrics

    def init_train_state(key: Key) -> TrainState:
        key, env_key = jax.random.split(key)
        train_state = init_fn(key)
        obs, env_state = utils.init_env_state(key=env_key, env=env, num_envs=num_envs, bc_indicator=bc_indicator)
        train_state = train_state.replace(last_obs=obs, last_env_state=env_state)
        return train_state

    # Define the training loop
    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(init_train_state)(jax.random.split(init_key, num_seeds))
        keys = jax.random.split(key, num_iterations)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=train_eval_loop_body, init=train_state, xs=keys)
        return state, metrics

    return jax.jit(scan_train_fn)


def make_loop_train_fn(
    # env : OfflineDatasetEnv(),
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    max_episode_steps: int,
    stochastic_eval: bool,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
    demo_path: str | None = None,
    bc_indicator: bool = False,
):
    from src.runners.maniskill_runner import (
        make_eval_fn as make_eval_fn,
        make_rollout_fn as make_rollout_fn,
    )

    train_log_interval = int((total_time_steps / (num_steps * num_envs)) // num_eval)

    train_log_interval = (
        int((total_time_steps / (num_steps * num_envs)) // num_eval) // 4
    )

    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    if rollout_fn is None:
        rollout_fn = make_rollout_fn(env, num_steps=num_steps, num_envs=num_envs, demo_path=demo_path, bc_indicator=bc_indicator)

    if eval_fn is None:
        eval_fn = make_eval_fn(eval_env, max_episode_steps, demo_path=demo_path, bc_indicator=bc_indicator)

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_eval
        train_steps_per_iteration = num_train_steps // num_iterations
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(last_obs=obs, last_env_state=None)
        logging.info(f"Starting training for {num_iterations} iterations.")
        logging.info(f"Train steps per iteration: {train_steps_per_iteration}.")
        logging.info(f"Total time steps: {total_time_steps}.")

        # Create checkpoint directories under src/ for saving model parameters
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        actor_checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'actor')
        critic_checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'critic')
        os.makedirs(actor_checkpoint_dir, exist_ok=True)
        os.makedirs(critic_checkpoint_dir, exist_ok=True)

        step = 0
        for i in range(num_iterations):
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key = jax.random.split(key, 3)
                # Collect trajectories from `state`
                policy = policy_fn(state, False)
                transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )
                # Execute an update to the policy with `transitions`
                state, train_metrics = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

                if step % train_log_interval == 0:
                    log_callback(state, utils.prefix_dict("train", train_metrics))
                step += 1
            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)
            eval_metrics = eval_fn(eval_key, policy)
            state = state.replace(iteration=state.iteration + 1)

            # Save model parameters after eval run
            # Helper function to recursively convert JAX state dict to torch tensors
            def jax_dict_to_torch(jax_dict, prefix=''):
                torch_dict = {}
                for k, v in jax_dict.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        torch_dict.update(jax_dict_to_torch(v, full_key))
                    elif isinstance(v, (jnp.ndarray, np.ndarray)):
                        try:
                            torch_dict[full_key] = torch.from_numpy(np.array(jax.device_get(v)))
                        except (TypeError, ValueError):
                            # Skip non-numeric types
                            pass
                return torch_dict
            
            # Actor
            jax_actor_dict = to_state_dict(state.actor.params)
            torch_actor_dict = jax_dict_to_torch(jax_actor_dict)
            torch.save(torch_actor_dict, os.path.join(actor_checkpoint_dir, f'{env.spec.id}_actor_step_{int(state.iteration)}_bc.pth'))
            
            # Critic
            jax_critic_dict = to_state_dict(state.critic.params)
            torch_critic_dict = jax_dict_to_torch(jax_critic_dict)
            torch.save(torch_critic_dict, os.path.join(critic_checkpoint_dir, f'{env.spec.id}_critic_step_{int(state.iteration)}_bc.pth'))
            
            log_callback(state, utils.prefix_dict("eval", eval_metrics))
        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn
