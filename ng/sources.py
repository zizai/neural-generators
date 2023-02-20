import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ng import au_const


class GenericSource(PyTreeNode):
    def __call__(self, r, t, *args, **kwargs):
        raise NotImplementedError


class ElectronSource(GenericSource):
    mu: chex.Array
    sigma: chex.Array
    p: chex.Array

    def sample(self, rng, n_samples):
        r = jax.random.normal(rng, (n_samples, 2))
        r = self.mu + r * self.sigma
        return r

    def __call__(self, r, t, *args, **kwargs):
        center_x = self.mu[..., 0]
        center_y = self.mu[..., 1]
        W_x = self.sigma[..., 0] ** 2
        W_y = self.sigma[..., 1] ** 2

        psi = jnp.exp(1j / au_const.hbar * jnp.sum(self.p * r, axis=-1, keepdims=True))
        psi *= (jnp.sqrt(jnp.exp(-(r[..., 0] - center_x) ** 2 / 2 / W_x) / jnp.sqrt(2 * jnp.pi) / W_x) *
                jnp.sqrt(jnp.exp(-(r[..., 1] - center_y) ** 2 / 2 / W_y) / jnp.sqrt(2 * jnp.pi) / W_y))

        rho = psi.conj() * psi
        rho_bar = jnp.mean(rho)
        psi /= jnp.sqrt(rho_bar)

        # assert np.allclose(np.mean(np.abs(psi) ** 2), 1.)

        return psi


class ContinuousPointSource(GenericSource):
    loc: chex.Array
    E0: chex.Array
    k0: chex.Array
    omega: chex.Array

    def sample(self, rng, n_samples=1):
        r = jnp.zeros((n_samples, 2)) + self.loc
        t = jnp.zeros((n_samples, 1))
        return r, t

    def __call__(self, r, t, *args, **kwargs):
        delta_r = jnp.alltrue(jnp.equal(r, self.loc), axis=-1, keepdims=True)
        return self.E0 * delta_r * jnp.exp(-1j * self.omega * t)


class GaussianPulseSource(GenericSource):
    loc: chex.Array
    w_y: chex.Scalar
    t0: chex.Scalar
    sigma_t: chex.Scalar
    E0: chex.Array
    k0: chex.Array
    omega: chex.Array

    def sample(self, rng, n_samples=100):
        key1, key2 = jax.random.split(rng)
        source_x = self.loc[:, 0]
        source_y = self.loc[:, 1]
        rx = jnp.zeros((n_samples, 1)) + source_x
        ry = jax.random.normal(key1, (n_samples, 1)) * self.w_y / jnp.sqrt(2) + source_y
        r = jnp.concatenate((rx, ry), -1)

        t = jax.random.normal(key2, (n_samples, 1)) * self.sigma_t + self.t0
        return r, t

    def __call__(self, r, t, *args, **kwargs):
        source_x = self.loc[:, 0]
        source_y = self.loc[:, 1]
        delta_r = jnp.alltrue(jnp.equal(r[..., 0:1], source_x), axis=-1, keepdims=True)
        envelope = delta_r * jnp.exp(-(r[..., 1:2] - source_y) ** 2 / self.w_y ** 2) * jnp.exp(-(t - self.t0) ** 2 / 2 / self.sigma_t ** 2)
        return self.E0 * envelope * jnp.exp(-1j * self.omega * t)
