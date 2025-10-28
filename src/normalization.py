import functools

import flax.struct as struct
import jax
import jax.numpy as jnp


class NormalizationState(struct.PyTreeNode):
    mean: struct.PyTreeNode
    var: struct.PyTreeNode
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
        M2 = m_a + m_b + jnp.square(delta) * state_count * batch_size / count
        new_var = M2 / count
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
    def normalize(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> struct.PyTreeNode:
        return jax.tree.map(
            lambda x, m, v: (x - m) / jnp.sqrt(v + 1e-2), tree, state.mean, state.var
        )
