import chex
import distrax
import jax
import jax.numpy as jnp

from flax import linen

from ng.dnb import DNBLayer


class NeuralGeneratorLayer(linen.Module):
    features: int
    model_type: str = 'mlp'

    @linen.compact
    def __call__(self, h, x):
        features = self.features

        if self.model_type == 'dnb':
            h, x = DNBLayer(features)(h, x)
        elif self.model_type == 'mlp':
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        return h, x
