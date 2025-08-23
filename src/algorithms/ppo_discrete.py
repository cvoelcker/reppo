import logging
import math
import time
import typing
from typing import Callable, Optional

import distrax
import gymnax
import hydra
import jax
import optax
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.experimental import checkify
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BatchEnv,
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
)
from src.algorithms import utils
from src.algorithms.normalization import NormalizationState, Normalizer

logging.basicConfig(level=logging.INFO)


## INITIALIZE CLASS STRUCTURES (NETWORKS, STATES, ...)
class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
        state: Optional[PyTreeNode] = None,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass


class PPOConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    lmbda: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    total_time_steps: int
    num_steps: int
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_advantages: bool
    normalize_env: bool
    anneal_lr: bool
    num_eval: int = 25
    max_episode_steps: int = 1000
    hidden_dim: int = 64


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]


class PPOTrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
    normalization_state: NormalizationState | None = None
    critic_normalization_state: NormalizationState | None = None


class PPONetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        def linear_layer(in_features, out_features, scale=jnp.sqrt(2)):
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                kernel_init=nnx.initializers.orthogonal(scale=scale),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        self.actor_module = nnx.Sequential(
            linear_layer(obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, action_dim, scale=0.01),
        )
        self.critic_module = nnx.Sequential(
            linear_layer(critic_obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, 1, scale=1.0),
        )

    def critic(self, obs: jax.Array) -> jax.Array:
        return self.critic_module(obs).squeeze()

    def actor(self, obs: jax.Array) -> distrax.Distribution:
        logits = self.actor_module(obs)
        pi = distrax.Categorical(logits=logits)
        return pi


def make_policy(train_state: PPOTrainState) -> Policy:
    normalizer = Normalizer()

    def policy(
        key: PRNGKey, obs: jax.Array, state: struct.PyTreeNode = None
    ) -> tuple[jax.Array, jax.Array]:
        if train_state.normalization_state is not None:
            obs = normalizer.normalize(train_state.normalization_state, obs)
        model = nnx.merge(train_state.graphdef, train_state.params)
        pi = model.actor(obs)
        value = model.critic(obs)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, dict(log_prob=log_prob, value=value)

    return policy


def make_eval_fn(
    env: Environment, max_episode_steps: int
) -> Callable[[jax.random.PRNGKey, Policy], dict[str, float]]:
    def evaluation_fn(key: jax.random.PRNGKey, policy: Policy):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            env_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                env_key, env_state, action.clip(-1.0 + 1e-4, 1.0 - 1e-4)
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key)
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


def make_init(
    cfg: PPOConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> PPOTrainState:
    def init(key: jax.random.PRNGKey) -> PPOTrainState:
        # Number of calls to train_step
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        # Number of calls to train_iter, add 1 if not divisible by eval_interval
        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, model_key = jax.random.split(key)
        # Intialize the model
        networks = PPONetworks(
            obs_dim=env.observation_space(env_params).shape[0],
            critic_obs_dim=env.observation_space(env_params).shape[0],
            action_dim=env.action_space(env_params).n,
            hidden_dim=cfg.hidden_dim,
            rngs=nnx.Rngs(model_key),
        )

        # Set initial learning rate
        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 1e-6, num_updates)

        # Initialize the optimizer
        if cfg.max_grad_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr),
            )
        else:
            optimizer = optax.adam(lr)

        # Reset and fully initialize the environment
        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, env_state = env.reset(env_key)
        # randomize initial time step to prevent all envs stepping in tandem
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        if cfg.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(obs)
            obs = normalizer.normalize(norm_state, obs)
        else:
            norm_state = None

        # Initialize the state observations of the environment
        return PPOTrainState.create(
            iteration=0,
            time_steps=0,
            graphdef=nnx.graphdef(networks),
            params=nnx.state(networks),
            tx=optimizer,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=obs,
            normalization_state=norm_state,
            critic_normalization_state=norm_state
        )

    return init


def make_train_fn(
    cfg: PPOConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[PPOTrainState, dict[str, jax.Array]], None] = None,
    num_seeds: int = 1,
):
    # Initialize the environment and wrap it to admit vectorized behavior.
    env_params = env_params or env.default_params
    env = BatchEnv(env)
    env = LogWrapper(env, cfg.num_envs)
    eval_fn = make_eval_fn(env, cfg.max_episode_steps)
    normalizer = Normalizer()
    eval_interval = int(
        (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
    )

    def collect_rollout(
        key: PRNGKey, train_state: PPOTrainState
    ) -> tuple[Transition, PPOTrainState]:
        model = nnx.merge(train_state.graphdef, train_state.params)

        # Take a step in the environment
        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry

            if cfg.normalize_env:
                norm_state = normalizer.update(train_state.normalization_state, obs)
                obs = normalizer.normalize(norm_state, obs)
                train_state = train_state.replace(normalization_state=norm_state)
                critic_obs = normalizer.normalize(
                    train_state.critic_normalization_state, critic_obs
                )
            # Select action
            key, act_key, step_key = jax.random.split(key, 3)
            pi = model.actor(obs)
            action = pi.sample(seed=act_key)
            # Take a step in the environment
            step_key = jax.random.split(step_key, cfg.num_envs)
            next_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action.clip(-1.0 + 1e-4, 1.0 - 1e-4)
            )
            # Record the transition
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                reward=reward,
                log_prob=pi.log_prob(action),
                value=model.critic(critic_obs),
                done=done,
                truncated=next_env_state.truncated,
                info=info,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        # Collect rollout via lax.scan taking steps in the environment
        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
        )
        # Aggregate the transitions across all the environments to reset for the next iteration
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def learn_step(
        key: PRNGKey, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        # Compute advantages and target values
        model = nnx.merge(train_state.graphdef, train_state.params)
        if cfg.normalize_env:
            last_critic_obs = normalizer.normalize(
                train_state.critic_normalization_state, train_state.last_critic_obs
            )
        else:
            last_critic_obs = train_state.last_critic_obs
        last_value = model.critic(last_critic_obs)

        def compute_advantage(carry, transition):
            gae, next_value = carry
            done = transition.done
            truncated = transition.truncated
            reward = transition.reward
            value = transition.value
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.lmbda * (1 - done) * gae
            truncated_gae = reward + cfg.gamma * next_value - value
            gae = jnp.where(truncated, truncated_gae, gae)
            return (gae, value), gae

        # Compute the advantage using GAE
        _, advantages = jax.lax.scan(
            compute_advantage,
            (jnp.zeros_like(last_value), last_value),
            batch,
            reverse=True,
        )
        target_values = advantages + batch.value

        data = (batch, advantages, target_values)
        # Reshape data to (num_steps * num_envs, ...)
        data = jax.tree.map(
            lambda x: x.reshape(
                (math.floor(cfg.num_steps * cfg.num_envs), *x.shape[2:])
            ),
            data,
        )

        def update(train_state, key) -> tuple[PPOTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, advantages, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )
                if cfg.normalize_advantages:
                    advantages = (advantages - jnp.mean(advantages)) / (
                        jnp.std(advantages) + 1e-8
                    )

                # Define the loss function
                def loss_fn(params):
                    model = nnx.merge(train_state.graphdef, params)
                    pi = model.actor(minibatch.obs)
                    value = model.critic(minibatch.critic_obs)
                    log_prob = pi.log_prob(minibatch.action)
                    value_pred_clipped = minibatch.value + (
                        value - minibatch.value
                    ).clip(-cfg.clip_ratio, cfg.clip_ratio)
                    value_error = jnp.square(value - target_values)
                    value_error_clipped = jnp.square(value_pred_clipped - target_values)
                    value_loss = 0.5 * jnp.mean(
                        (1.0 - minibatch.truncated)
                        * jnp.maximum(value_error, value_error_clipped)
                    )

                    ratio = jnp.exp(log_prob - minibatch.log_prob)
                    checkify.check(
                        jnp.allclose(ratio, 1.0) | (idx != 1),
                        debug=True,
                        msg="Ratio not equal to 1 on first iteration: {r}",
                        r=ratio,
                    )

                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                        * advantages
                    )
                    actor_loss = -jnp.mean(
                        (1.0 - minibatch.truncated)
                        * jnp.minimum(actor_loss1, actor_loss2)
                    )
                    entropy_loss = jnp.mean(pi.entropy())

                    loss = (
                        actor_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_loss
                    )

                    return loss, dict(
                        actor_loss=actor_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        loss=loss,
                        mean_value=value.mean(),
                        mean_log_prob=log_prob.mean(),
                        mean_advantages=advantages.mean(),
                        mean_action=minibatch.action.mean(),
                        mean_reward=minibatch.reward.mean(),
                    )

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                output, grads = grad_fn(train_state.params)

                # Global gradient norm (all parameters combined)
                flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
                global_grad_norm = jnp.linalg.norm(flat_grads)

                metrics = output[1]
                metrics["advantages"] = advantages
                metrics["global_grad_norm"] = global_grad_norm
                train_state = train_state.apply_gradients(grads)
                return (idx + 1, train_state), metrics

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)

            mini_batch_size = (
                math.floor(cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            )
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics = jax.lax.scan(
            f=update,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    # Define the training loop
    def train_fn(key: PRNGKey) -> tuple[PPOTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: PPOTrainState, key: PRNGKey
            ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                # Collect trajectories from `state`
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
                # Execute an update to the policy with `transitions`
                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics, **update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            train_state, train_metrics = jax.lax.scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            policy = make_policy(train_state)
            eval_metrics = eval_fn(eval_key, policy)
            metrics = {
                "time_step": train_state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
            }

            return train_state, metrics

        def loop_body(
            train_state: PPOTrainState, key: PRNGKey
        ) -> tuple[PPOTrainState, dict]:
            # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )
            jax.debug.callback(log_callback, train_state, metrics)
            return train_state, metrics

        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    return train_fn


def plot_history(history: list[dict[str, jax.Array]]):
    steps = jnp.array([m["time_step"][0] for m in history])
    eval_return = jnp.array([m["eval/episode_return"].mean() for m in history])
    eval_return_std = jnp.array([m["eval/episode_return"].std() for m in history])
    fig = go.Figure(
        [
            go.Scatter(
                x=steps,
                y=eval_return,
                name="Mean Episode Return",
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return + eval_return_std,
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return - eval_return_std,
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(50, 127, 168, 0.3)",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        xaxis=dict(title=dict(text="Environment Steps")),
    )

    return fig


def run(cfg: DictConfig):
    metric_history = []

    # Define callback to log metrics during training
    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        # Use pop() with a default value of None in case 'advantages' key doesn't exist
        advantages = metrics.pop("train/advantages", None)
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "train/advantages": wandb.Histogram(advantages),
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        # Push log data to WandB
        wandb.log(log_data, step=state.time_steps[0])

    logging.info(OmegaConf.to_yaml(cfg))

    # Set up the experimental environment
    if cfg.env.type == "gymnax":
        env, env_params = gymnax.make(cfg.env.name)
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    key = jax.random.PRNGKey(cfg.seed)
    train_fn = make_train_fn(
        cfg=PPOConfig(**cfg.hyperparameters),
        env=env,
        env_params=env_params,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
    )
    for i in range(cfg.trials):
        # Initialize WandB reporting
        key, train_key = jax.random.split(key)
        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
            config=OmegaConf.to_container(cfg),
            name=f"ppo-{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )
        start = time.perf_counter()
        train_state, metrics = jax.jit(train_fn)(train_key)
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        # jnp.savez("metrics.npz", **metrics) # TODO: fix the directory here to save to a unique output directory
        wandb.finish()


def tune(cfg: DictConfig):
    def log_callback(state, metrics):
        episode_return = metrics["eval/episode_return"].mean()
        t = state.time_steps[0]
        wandb.log(
            {
                "episode_return": episode_return,
            },
            step=t,
        )

    env = MjxGymnaxWrapper(cfg.env.name, episode_length=cfg.env.max_episode_steps)

    def train_agent():
        wandb.init(project=cfg.wandb.project)
        run_cfg = OmegaConf.to_container(cfg)
        for k, v in dict(wandb.config).items():
            run_cfg["experiment"]["hyperparameters"][k] = v
        ppo_cfg = PPOConfig(**run_cfg["experiment"]["hyperparameters"])
        train_fn = make_train_fn(
            cfg=ppo_cfg,
            env=env,
            log_callback=log_callback,
            num_seeds=cfg.num_seeds,
        )
        train_fn = jax.jit(train_fn)
        logging.info(f"Running experiment with params: \n {run_cfg}")
        key = jax.random.PRNGKey(cfg.seed)
        train_state, metrics = train_fn(key)
        jax.block_until_ready(metrics)

    sweep_id = wandb.sweep(
        sweep={
            "name": f"{cfg.name}-{cfg.env.name}",
            "method": "bayes",
            "metric": {"name": "episode_return", "goal": "maximize"},
            "parameters": {
                "lr": {
                    "values": [1e-4, 3e-4, 1e-3],
                },
                "normalize_env": {
                    "values": [True, False],
                },
            },
        },
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
    )
    wandb.agent(sweep_id, function=train_agent, count=cfg.tune.num_runs)


@hydra.main(version_base=None, config_path="../../config", config_name="ppo")
def main(cfg: DictConfig):
    if cfg.tune:
        tune(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
