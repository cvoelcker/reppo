import functools

import flax.struct as struct
import jax
import jax.numpy as jnp


class NormalizationState(struct.PyTreeNode):
    mean: struct.PyTreeNode
    var: struct.PyTreeNode
    mean_return: jax.Array
    var_return: jax.Array
    count: int


class Normalizer:
    @functools.partial(jax.jit, static_argnums=0)
    def init(self, tree: struct.PyTreeNode) -> NormalizationState:
        """
        Initialize the normalization state. Pytree should be a single instance without batch dimensions.
        """
        return NormalizationState(
            mean=jax.tree.map(lambda x: jnp.zeros_like(x), tree),
            var=jax.tree.map(lambda x: jnp.ones_like(x), tree),
            count=jax.tree.map(lambda x: jnp.array(0, dtype=jnp.int32), tree),
            mean_return=jnp.array(0.0),
            var_return=jnp.array(1.0)
        )

    def _compute_stats(self, state_mean, state_var, state_count, obs: jax.Array):
        var = jnp.var(obs, axis=0)
        mean = jnp.mean(obs, axis=0)
        batch_size = obs.shape[0]
        delta = mean - state_mean
        count = state_count + batch_size
        new_mean = state_mean + delta * batch_size / count
        m_a = state_var * state_count
        m_b = var * batch_size
        m2 = m_a + m_b + jnp.square(delta) * state_count * batch_size / count
        new_var = m2 / count
        return new_mean, new_var, count

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> NormalizationState:
        tree = jax.tree.map(lambda x, m: x.reshape(-1, *m.shape), tree, state.mean)
        stats = jax.tree.map(
            lambda m, v, c, x: self._compute_stats(m, v, c, x),
            state.mean,
            state.var,
            state.count,
            tree,
        )
        mean, var, count = jax.tree.transpose(
            jax.tree.structure(tree), jax.tree.structure(("*", "*", "*")), stats
        )
        return state.replace(
            mean=mean,
            var=var,
            count=count,
        )
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_returns(
        self, state: NormalizationState, returns: jax.Array
    ) -> NormalizationState:
        batch_size = returns.shape[1]
        var = jnp.var(returns)
        mean = jnp.mean(returns)
        delta = mean - state.mean_return
        new_mean = state.mean_return + delta * batch_size / state.count
        m_a = state.var_return * state.count
        m_b = var * batch_size
        m2 = m_a + m_b + jnp.square(delta) * state.count * batch_size / state.count
        new_var = m2 / state.count
        jax.debug.print("Updated return normalizer: mean={}, var={}", new_mean, new_var)
        return state.replace(
            mean_return=new_mean,
            var_return=new_var,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def normalize(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> struct.PyTreeNode:
        return jax.tree.map(
            lambda x, m, v: (x - m) / jnp.sqrt(v + 1e-8), tree, state.mean, state.var
        )
    
    @functools.partial(jax.jit, static_argnums=0)
    def normalize_returns(
        self, state: NormalizationState, rewards: jax.Array
    ) -> jax.Array:
        return (rewards - state.mean_return) / jnp.sqrt(state.var_return + 1e-8)

    @functools.partial(jax.jit, static_argnums=0)
    def denormalize_returns(
        self, state: NormalizationState, normed_rewards: jax.Array
    ) -> jax.Array:
        return normed_rewards * jnp.sqrt(state.var_return + 1e-8) + state.mean_return
