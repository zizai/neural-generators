import chex
import distrax
import jax
import jax.numpy as jnp

from flax import linen

from ng.dnb import DynamicNeuralBasis


class NeuralGeneratorLayer(linen.Module):
    features: int

    @linen.compact
    def __call__(self, h, x):
        features = self.features
        heads = int(features // 32)

        x0 = x
        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x) + x0

        # h = linen.silu(linen.Dense(features)(h))
        # h = linen.Dense(features)(h) + h
        # x = DynamicNeuralBasis(heads, features)(x, h) + x
        # # x = linen.LayerNorm()(x)
        # x0 = x
        # x = linen.silu(linen.Dense(features)(x))
        # x = linen.Dense(features)(x) + x0
        # # x = linen.LayerNorm()(x)
        return h, x
