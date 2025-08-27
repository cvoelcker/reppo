import typing
from typing import Callable, Optional, TypeVar

import gymnasium
from gymnax import EnvState
import jax
from flax import struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams
from jax.random import PRNGKey
from flax import nnx
import jax.numpy as jnp
import numpy as np

from src.algorithms import utils

Config = TypeVar("Config", bound=struct.PyTreeNode)
Key = jax.Array
Metrics = dict[str, jax.Array]


@struct.dataclass
class TrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array


class Transition(struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    truncated: jax.Array
    extras: dict[str, jax.Array]


class Policy(typing.Protocol):
    def __call__(
        self,
        key: Key,
        obs: PyTreeNode,
        state: Optional[PyTreeNode] = None,
    ) -> tuple[jax.Array, Optional[PyTreeNode]]:
        pass


class InitFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
    ) -> TrainState:
        pass


class LearnerFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        train_state: TrainState,
        batch: Transition,
    ) -> tuple[TrainState, Metrics]:
        pass


class RolloutFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        train_state: TrainState,
        policy: Policy,
    ) -> tuple[Transition, TrainState]:
        pass


class EvalFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
        policy: Policy,
    ) -> dict[str, jax.Array]:
        pass


class PolicyFn(typing.Protocol):
    def __call__(
        self,
        train_state: TrainState,
        eval_mode: bool = False,
    ) -> Policy:
        pass


class LogCallback(typing.Protocol):
    def __call__(
        self,
        train_state: TrainState,
        metrics: dict[str, jax.Array],
    ) -> None:
        pass


class TrainFn(typing.Protocol):
    def __call__(
        self,
        key: Key,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        pass


def make_train_fn(
    cfg: Config,
    env: Environment | tuple[Environment, Environment],
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    eval_fn: EvalFn | None = None,
    rollout_fn: RolloutFn | None = None,
    log_callback: LogCallback | None = None,
    num_seeds: int = 1,
    mode: str = "scan",
) -> TrainFn:
    # Initialize the environment and wrap it to admit vectorized behavior.
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    eval_interval = int(
        (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
    )

    if eval_fn is None:
        eval_fn = make_eval_fn(eval_env, cfg.max_episode_steps)

    if rollout_fn is None:
        rollout_fn = make_rollout_fn(cfg, env)

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
        policy = policy_fn(train_state, True)
        eval_metrics = eval_fn(eval_key, policy)
        metrics = {
            "time_step": train_state.time_steps,
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
        obs, env_state = utils.init_env_state(
            key=env_key, env=env, num_envs=cfg.num_envs
        )
        train_state = train_state.replace(last_obs=obs, last_env_state=env_state)
        return train_state

    # Define the training loop
    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(init_train_state)(jax.random.split(init_key, num_seeds))
        keys = jax.random.split(key, num_iterations)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=train_eval_loop_body, init=train_state, xs=keys)
        return state, metrics

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(
            last_obs=jax.tree.map(jnp.array, obs), last_env_state=None
        )
        for i in range(num_iterations):
            for _ in range(eval_interval):
                print(f"Iteration {i}, Eval Step {_}")
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

            key, eval_key = jax.random.split(key)
            eval_metrics = eval_fn(eval_key, policy)
            metrics = {
                "time_step": state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
            }
            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, metrics)
        return state, metrics

    if mode == "loop":
        return loop_train_fn
    else:
        return scan_train_fn


def make_eval_fn(env: Environment, max_episode_steps: int) -> EvalFn:
    def evaluation_fn(key: Key, policy: Policy):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            env_key = jax.random.split(env_key, env.num_envs)
            obs, env_state, reward, done, info = env.step(env_key, env_state, action)
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, env_state = env.reset(init_key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        return {
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            ),
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
        }

    return evaluation_fn


def make_rollout_fn(cfg: Config, env: Environment) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment
        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs = carry

            # Select action
            key, act_key, step_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            step_key = jax.random.split(step_key, cfg.num_envs)
            next_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            # Record the transition
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                truncated=next_env_state.truncated,
                extras=info,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
            ), transition

        # Collect rollout via lax.scan taking steps in the environment
        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
            ),
            length=cfg.num_steps,
        )
        # Aggregate the transitions across all the environments to reset for the next iteration
        _, last_env_state, train_state, last_obs = rollout_state

        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    return collect_rollout


def make_gymnasium_rollout_fn(cfg: Config, env: gymnasium.Env) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        # Take a step in the environment

        transitions = []
        obs = train_state.last_obs
        for _ in range(cfg.num_steps):
            # Select action
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(
                np.array(action)  # Gymnasium envs don't support vectorized actions
            )
            # Record the transition
            transition = Transition(
                obs=jnp.array(obs),
                action=jnp.array(action),
                reward=jnp.array(reward),
                done=jnp.array(done),
                truncated=jnp.array(truncated),
                extras={},
            )
            transitions.append(transition)
            obs = next_obs

        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            last_env_state=None,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )
        return transitions, train_state

    return collect_rollout


def make_gymnasium_eval_fn(cfg: Config, env: gymnasium.Env) -> EvalFn:
    def evaluate(key: Key, policy: Policy) -> dict:
        # Evaluate the policy in the environment
        key, eval_key = jax.random.split(key)

        # Reset the environment
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        running_reward = np.zeros(env.num_envs)
        dones = []
        num_episodes = 0
        for _ in range(cfg.max_episode_steps + 1):
            # Select action
            action, _ = policy(eval_key, obs)
            # Step the environment
            next_obs, reward, done, truncated, info = env.step(np.array(action))
            dones.append(done)
            obs = next_obs
            total_reward += np.where(done, running_reward + reward, 0)
            running_reward = np.where(done, 0, running_reward + reward)
          
            if "final_info" in info:
                for info in info["final_info"]:
                    if info and "episode" in info:
                        total_reward += info["episode"]["r"]
                        num_episodes += 1
                     

        total_reward += np.sum(running_reward)

        return {
            "episode_return": total_reward / num_episodes,
            "num_episodes": num_episodes,
        }

    return evaluate
