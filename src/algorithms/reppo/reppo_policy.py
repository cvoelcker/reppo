from typing import Callable
from dataclasses import dataclass

import jax
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space, Box
from src.algorithms.reppo.common import REPPOTrainState
from src.common import Policy

from src.normalization import Normalizer
import distrax


### Vanilla SAC style Policy


class REPPOPolicy(nnx.Module):
    def __init__(
        self,
        base: nnx.Module,
        normalizer: Normalizer | None,
        normalization_state,
        eval: bool,
        action_space: Space,
    ):
        self.base = base
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self._eval_mode = eval
        self.action_space = action_space

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        if self._eval_mode:
            action = self.base.det_action(x)
        else:
            pi = self.base(x, **kwargs)
            action = pi.sample(seed=key)
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}


@dataclass
class LangevinConfig:
    # params for the step size (epsilon) computation
    a: float = 1.0
    b: float = 50.0
    eps_const: bool = (
        False  # if we keep the epsilon the same across planning iterations
    )

    grad_step_size: float = 1.0
    scale_noise_by_act_dim: bool = False

    # MPC params
    mpc_iterations: int = 5
    num_mpc_samples: int = 1
    num_policy_mpc_samples: int = 1
    mpc_temperature: float = 0.5

    gamma: float = 0.99

    exploration_noise_sigma: float = 0.3


class LangevinPolicy(nnx.Module):
    def __init__(
        self,
        actor: nnx.Module,
        critic: nnx.Module,
        normalizer: Normalizer | None,
        normalization_state,
        action_space: Space,
        config: LangevinConfig = LangevinConfig(),
        eval: bool = False,
    ):
        self.actor = actor
        self.critic = critic
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self.action_space = action_space
        self.config = config
        self._eval_mode = eval

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        alpha = self.actor.temperature()

        # if self._eval_mode:
        #     action = self.actor.det_action(x)
        #     if isinstance(self.action_space, Box):
        #         action = action.clip(-0.999, 0.999)
        #     return action, {}

        action = get_langevin_action(
            self.actor,
            self.critic,
            x,
            # eta_scaler,
            alpha,
            key,
            self.config,
        )
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}


def get_grad_action(config: LangevinConfig, actor, critic, obs, actions, alpha):
    actions = jnp.clip(actions, -1 + 1e-4, 1 - 1e-4)

    def get_values(a, x):
        # if not h.uniform_action_prior:
        #     act = jnp.tanh(act)
        vals = critic(x, a)["value"]
        return vals.mean()

    def get_log_probs(a):
        return actor.log_prob(a)

    val_grad = jax.grad(get_values)
    val_grad = jax.vmap(jax.vmap(val_grad), in_axes=[0, None])(actions, obs)

    prior_grad = jax.vmap(jax.jacrev(get_log_probs))(actions)
    # jax.debug.print("Prior grad off diagonal: {}", prior_grad[:, 0, 1])
    prior_grad = jnp.diagonal(prior_grad, axis1=1, axis2=2)
    prior_grad = jnp.transpose(prior_grad, (0, 2, 1))

    return val_grad, prior_grad


def epsilon(config: LangevinConfig, it: int) -> float:
    return config.a / (config.b + it * (not config.eps_const))


def get_langevin_action(
    actor,
    critic,
    state: jax.Array,
    # eta_scaler: jax.Array,
    alpha: jax.Array,
    rand_key: jax.Array,
    config: LangevinConfig = LangevinConfig(),
):
    rand_key, act_key = jax.random.split(rand_key)

    rand_key, act_key = jax.random.split(rand_key)
    pi = actor(state)
    actor_action, log_probs = pi.sample_and_log_prob(
        sample_shape=(config.num_policy_mpc_samples,), seed=act_key
    )

    mean = actor_action.mean(axis=0)
    std = jnp.ones_like(mean) * config.exploration_noise_sigma

    random_actions = (
        mean
        + jax.random.normal(
            act_key,
            shape=[config.num_mpc_samples - config.num_policy_mpc_samples, *mean.shape],
        )
        * std
    )

    actions = jnp.concatenate([actor_action, random_actions], axis=0)
    actions = jnp.clip(actions, -1 + 1e-4, 1 - 1e-4)
    
    values = critic(state[None].repeat(config.num_mpc_samples, axis=0), actions)[
        "value"
    ]
    log_probs = jax.vmap(pi.log_prob)(actions)

    # jax.debug.print("Initial action value: {}", values.mean(), ordered=True)
    # jax.debug.print(
    #     "Initial action mean logits {}", log_probs.mean(), ordered=True
    # )

    def _mpc_step(carry, i):
        actions, rand_key = carry
        eta_key, soft_key, rand_key = jax.random.split(rand_key, 3)

        val_grad, lp_grad  = get_grad_action(
            config, pi, critic, state, actions, alpha
        )

        # eta = jnp.sqrt(epsilon(config, i)) * eta_scaler * jax.random.normal(eta_key, shape=actions.shape)
        eta = (
            jnp.sqrt(epsilon(config, i))
            * alpha
            * jax.random.normal(eta_key, shape=actions.shape)
        )
        if config.scale_noise_by_act_dim:
            eta = eta / (actions.shape[-1] ** 0.5)

        # action_delta = 0.5 * epsilon(config, i) * (val_grad * config.grad_step_size) + eta
        action_delta = 0.5 * epsilon(config, i) * ((val_grad + alpha * 0.001 * lp_grad) * config.grad_step_size) + eta
        # jax.debug.print("Langevin step {}/{}: eps = {}, max|grad| = {}, max|eta| = {}, max|delta| = {}", i+1, config.mpc_iterations, epsilon(config, i), jnp.abs(grad).max(), jnp.abs(eta).max(), jnp.abs(action_delta).max())
        actions = actions + action_delta
        actions = jnp.clip(actions, -1.0 + 1e-4, 1.0 - 1e-4)

        # jax.debug.print("Relative step size -- norm action_delta {}, norm eta {}, norm val_grad {}, norm lp_grad {}", jnp.linalg.norm(action_delta, axis=-1).mean(), jnp.linalg.norm(eta, axis=-1).mean(), jnp.linalg.norm(val_grad, axis=-1).mean(), jnp.linalg.norm(alpha * lp_grad, axis=-1).mean(), ordered=True)
        return (actions, rand_key), actions

    (_actions, _), _ = jax.lax.scan(
        _mpc_step, (actions, rand_key), jnp.arange(config.mpc_iterations)
    )
    if config.mpc_iterations > 1:
        actions = _actions
    values = critic(state[None].repeat(config.num_mpc_samples, axis=0), actions)[
        "value"
    ]
    log_probs = jax.vmap(pi.log_prob)(actions)
    top_softmax = jax.nn.softmax((values + alpha * log_probs) / config.mpc_temperature, axis=0)
    top = jax.vmap(
        lambda p: jax.random.choice(act_key, jnp.arange(actions.shape[0]), p=p),
        in_axes=[1],
    )(top_softmax)
    # jax.debug.print("Final top action value: {}", values[top, jnp.arange(state.shape[0])].mean(), ordered=True)
    # jax.debug.print("Final action mean logits {}", log_probs[top, jnp.arange(state.shape[0])].mean(), ordered=True)
    action = actions[top, jnp.arange(state.shape[0])]
    return action


def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = cfg.algorithm
    offset = None

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer()
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        if cfg.policy_method == "langevin":
            policy = LangevinPolicy(
                actor=actor_model,
                critic=critic_model,
                normalizer=normalizer if cfg.normalize_env else None,
                normalization_state=train_state.normalization_state,
                action_space=action_space,
                eval=eval,
            )
        else:
            policy = REPPOPolicy(
                base=actor_model,
                normalizer=normalizer if cfg.normalize_env else None,
                normalization_state=train_state.normalization_state,
                eval=eval,
                action_space=action_space,
            )
        policy.eval()

        return policy

    return policy_fn
