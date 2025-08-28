import logging
import math
import time
from typing import Callable

import hydra
import jax
import optax
from flax import nnx, struct
from flax.struct import PyTreeNode
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf
from gymnax.environments.spaces import Space
import wandb
from src.common import (
    Key,
    Policy,
    TrainState,
    Transition,
)
from src.normalization import Normalizer
from src.algorithms import envs, utils
from src.algorithms.reppo.networks import (
    CategoricalCriticNetwork,
    CriticNetwork,
    SACActorNetworks,
)

logging.basicConfig(level=logging.INFO)


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    reverse_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    action_size_target: float = 0
    reward_scale: float = 1.0


class REPPOTrainState(TrainState):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    normalization_state: PyTreeNode | None = None


def make_reppo_policy_fn(cfg: ReppoConfig) -> Callable[[REPPOTrainState, bool], Policy]:
    offset = (
        jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
        * (cfg.exploration_noise_max - cfg.exploration_noise_min)
        / (cfg.num_envs - cfg.exploration_base_envs)
    ) + cfg.exploration_noise_min

    offset = jnp.concatenate(
        [
            jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
            offset,
        ],
        axis=0,
    )

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer()
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        def policy(key: Key, obs: jax.Array, **kwargs) -> tuple[jax.Array, dict]:
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)

            if eval:
                action: jax.Array = actor_model.det_action(obs)
            else:
                pi = actor_model.actor(obs, scale=offset)
                action = pi.sample(seed=key)
                action = jnp.clip(action, -0.999, 0.999)
            return action, {}

        return policy

    return policy_fn


def make_reppo_init_fn(
    cfg: ReppoConfig,
    observation_space: Space,
    action_space: Space,
) -> Callable[[jax.Array], REPPOTrainState]:
    def init(key: Key) -> REPPOTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        actor_networks = SACActorNetworks(
            obs_dim=observation_space.shape[0],
            action_dim=action_space.shape[0],
            hidden_dim=cfg.actor_hidden_dim,
            ent_start=cfg.ent_start,
            kl_start=cfg.kl_start,
            use_norm=cfg.use_actor_norm,
            layers=cfg.num_actor_layers,
            use_skip=cfg.use_actor_skip,
            rngs=nnx.Rngs(model_key),
        )

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=observation_space.shape[0],
                action_dim=action_space.shape[0],
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=observation_space.shape[0],
                action_dim=action_space.shape[0],
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr)
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr)
            )
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )

        if cfg.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(jnp.zeros(observation_space.shape))
            # obs = normalizer.normalize(norm_state, obs)
        else:
            norm_state = None

        return REPPOTrainState.create(
            graphdef=actor_trainstate.graphdef,
            params=actor_trainstate.params,
            tx=actor_trainstate.tx,
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            normalization_state=norm_state,
            last_env_state=None,
            last_obs=None,
        )

    return init


def make_reppo_learner_fn(cfg: ReppoConfig):
    normalizer = Normalizer()

    def critic_loss_fn(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_pred = critic_model.critic_cat(minibatch.obs, minibatch.action).squeeze()

        target_values = minibatch.extras["target_values"]

        if cfg.hl_gauss:
            target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
                target_values, cfg.num_bins, cfg.vmin, cfg.vmax
            )
            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        else:
            critic_update_loss = optax.squared_error(
                critic_pred.reshape(-1, 1),
                target_values.reshape(-1, 1),
            )

        # Aux loss
        _, pred, pred_rew, value = critic_model.forward(minibatch.obs, minibatch.action)
        aux_loss = optax.squared_error(pred, minibatch.extras["next_emb"])
        aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1))
            * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
        )

        # compute l2 error for logging
        critic_loss = optax.squared_error(
            value,
            target_values,
        )
        critic_loss = jnp.mean(critic_loss)
        loss = jnp.mean(
            (1.0 - minibatch.truncated)
            * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
        )
        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=aux_rew_loss,
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
        )

    def actor_loss(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_target_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_model = nnx.merge(train_state.actor.graphdef, params)
        actor_target_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor_target.params
        )

        # SAC actor loss
        pi = actor_model.actor(minibatch.obs)
        pred_action, log_prob = pi.sample_and_log_prob(
            seed=minibatch.extras["action_key"]
        )
        value = critic_target_model.critic(minibatch.obs, pred_action)
        entropy = -log_prob

        # policy KL constraint
        if cfg.reverse_kl:
            pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                sample_shape=(16,), seed=minibatch.extras["kl_key"]
            )
            pi_action = jnp.clip(pi_action, -1 + 1e-4, 1 - 1e-4)

            old_pi = actor_target_model.actor(minibatch.obs)

            old_pi_act_log_prob = old_pi.log_prob(pi_action).mean(0)
            pi_act_log_prob = pi_act_log_prob.mean(0)
            kl = pi_act_log_prob - old_pi_act_log_prob
        else:
            old_pi_action, old_pi_act_log_prob = actor_target_model.actor(
                minibatch.obs
            ).sample_and_log_prob(sample_shape=(16,), seed=minibatch.extras["kl_key"])
            old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)

            old_pi_act_log_prob = old_pi_act_log_prob.mean(0)
            pi_act_log_prob = pi.log_prob(old_pi_action).mean(0)

            kl = old_pi_act_log_prob - pi_act_log_prob

        lagrangian = actor_model.lagrangian()

        if cfg.actor_kl_clip_mode == "full":
            actor_loss = (
                log_prob * jax.lax.stop_gradient(actor_model.temperature())
                - value
                + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
            )
        elif cfg.actor_kl_clip_mode == "clipped":
            actor_loss = jnp.where(
                kl < cfg.kl_bound,
                log_prob * jax.lax.stop_gradient(actor_model.temperature()) - value,
                kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
            )
        elif cfg.actor_kl_clip_mode == "value":
            actor_loss = (
                log_prob * jax.lax.stop_gradient(actor_model.temperature()) - value
            )
        else:
            raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

        # SAC target entropy loss
        target_entropy = cfg.action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(
            target_entropy
        )

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - cfg.kl_bound)

        # total loss
        loss = jnp.mean(actor_loss)
        if cfg.update_entropy_lagrangian:
            loss += jnp.mean(target_entropy_loss)
        if cfg.update_kl_lagrangian:
            loss += jnp.mean(lagrangian_loss)

        return loss, dict(
            actor_loss=actor_loss,
            loss=loss,
            temp=actor_model.temperature(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean(),
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            entropy=entropy,
            entropy_loss=target_entropy_loss,
            target_values=minibatch.extras["target_values"].mean(),
        )

    def update(train_state: REPPOTrainState, batch: Transition):
        # Sample data at indices from the batch
        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
        critic_train_state = train_state.critic.apply_gradients(grads)
        train_state = train_state.replace(
            critic=critic_train_state,
        )
        critic_metrics = output[1]

        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
        actor_train_state = train_state.actor.apply_gradients(grads)
        train_state = train_state.replace(
            actor=actor_train_state,
        )
        actor_metrics = output[1]
        return train_state, {
            **critic_metrics,
            **actor_metrics,
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        mini_batch_size = (
            math.floor(cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
        )
        indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size, *x.shape[1:])),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)
        minibatches.extras["action_key"] = jax.random.split(
            act_key, cfg.num_mini_batches
        )
        minibatches.extras["kl_key"] = jax.random.split(kl_key, cfg.num_mini_batches)

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)
        # Compute mean metrics across mini-batches
        metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return train_state, metrics

    def nstep_lambda(batch: Transition):
        def loop(carry: tuple[jax.Array, ...], transition: Transition):
            lambda_return, truncated, importance_weight = carry

            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.extras["soft_reward"]
            value = transition.extras["value"]
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.extras["importance_weight"],
            ), lambda_return

        _, target_values = jax.lax.scan(
            f=loop,
            init=(
                batch.extras["value"][-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.extras["importance_weight"][0]),
            ),
            xs=batch,
            reverse=True,
        )
        return target_values

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        last_obs = train_state.last_obs
        if cfg.normalize_env:
            last_obs = normalizer.normalize(train_state.normalization_state, last_obs)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        emb, _, _, value = critic_model.forward(batch.obs, batch.action)
        og_pi = actor_model.actor(batch.obs)
        pi = actor_model.actor(batch.obs, scale=offset)
        key, act_key = jax.random.split(key)

        last_action, last_log_prob = actor_model.actor(last_obs).sample_and_log_prob(
            seed=act_key
        )
        last_emb, _, _, last_value = critic_model.forward(last_obs, last_action)

        log_probs = pi.log_prob(batch.action)
        next_log_prob = jnp.concatenate([log_probs[1:], last_log_prob[None]], axis=0)

        soft_reward = (
            batch.reward
            - cfg.gamma * next_log_prob * actor_model.temperature()
        )

        raw_importance_weight = jnp.nan_to_num(
            og_pi.log_prob(batch.action) - pi.log_prob(batch.action),
            nan=jnp.log(cfg.lmbda_min),
        )
        importance_weight = jnp.clip(
            raw_importance_weight, min=jnp.log(cfg.lmbda_min), max=jnp.log(1.0)
        )

        extras = {
            "soft_reward": soft_reward * cfg.reward_scale,
            "value": jnp.concatenate([value[1:], last_value[None]], axis=0),
            "next_emb": jnp.concatenate([emb[1:], last_emb[None]], axis=0),
            "importance_weight": importance_weight,
        }
        return extras

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        if cfg.normalize_env:
            new_norm_state = normalizer.update(
                train_state.normalization_state, batch.obs
            )
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs)
            )
            train_state = train_state.replace(normalization_state=new_norm_state)

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)

        batch.extras["target_values"] = nstep_lambda(batch=batch)

        # Reshape data to (num_steps * num_envs, ...)

        batch = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), batch
        )
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch),
            init=train_state,
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    return learner_fn


@hydra.main(version_base=None, config_path="../../../config", config_name="reppo")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
        config=OmegaConf.to_container(cfg),
        name=f"reppo-{cfg.name}-{cfg.env.name.lower()}",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)

    # Set up the experimental environment
    env_setup = envs.make_env(cfg)
    config = ReppoConfig(**cfg.hyperparameters)
    config = config.replace(
        action_size_target=(
            jnp.prod(jnp.array(env_setup.action_space.shape))
            * config.ent_target_mult
        )
    )
    if cfg.runner.type == "gymnax":
        from src.runners.gymnax_train_fn import make_train_fn
    elif cfg.runner.type == "gymnasium":
        from src.runners.gymnasium_train_fn import make_train_fn
    else:
        raise ValueError("Unknown environment type")

    env_setup = envs.make_env(cfg)
    train_fn = make_train_fn(
        cfg=config,
        env=(env_setup.env, env_setup.eval_env),
        init_fn=make_reppo_init_fn(
            cfg=config,
            observation_space=env_setup.observation_space,
            action_space=env_setup.action_space,
        ),
        learner_fn=make_reppo_learner_fn(config),
        policy_fn=make_reppo_policy_fn(config),
        log_callback=utils.make_log_callback(),
    )
    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()
