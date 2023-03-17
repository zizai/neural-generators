import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode


class GenericMaterial(PyTreeNode):
    def __call__(self, r, *args, **kwargs):
        raise NotImplementedError


class DielectricEllipse(GenericMaterial):
    rx: chex.Scalar
    ry: chex.Scalar
    x0: chex.Scalar = 0.
    y0: chex.Scalar = 0.
    eps_max: chex.Scalar = 9.

    def sample(self, rng, n_samples):
        r = jax.random.normal(rng, (n_samples, 2))
        r = jnp.stack([(r[..., 0] + self.x0) * self.rx, (r[..., 1] + self.y0) * self.ry], -1)
        return r

    def __call__(self, r, *args, **kwargs):
        eps_r = jnp.ones(r.shape)

        x, y = r[..., 0], r[..., 1]
        ellipse = ((x - self.x0) / self.rx) ** 2 + ((y - self.y0) / self.ry) ** 2 <= 1
        eps_r = jnp.where(ellipse[:, None], self.eps_max, eps_r)
        return eps_r


class DielectricVacuum(GenericMaterial):
    eps_max: chex.Scalar = 1.

    def __call__(self, r, *args, **kwargs):
        return jnp.ones(r.shape)
