import math
from functools import partial

import chex
import distrax
import jax
import jax.numpy as jnp
import typing
from flax import linen, struct

from ng import au_const
from ng.materials import GenericMaterial, DielectricVacuum
from ng.ng_layer import NeuralGeneratorLayer
from ng.potentials import GenericPotential
from ng.siren import SIREN
from ng.sources import CWSource
from ng.train_state import TrainState


class MaxwellPropagatorModelConfig(struct.PyTreeNode):
    t_domain: typing.Tuple
    x_domain: typing.Tuple
    y_domain: typing.Tuple
    dt: float
    sample_length: int
    c: float = 1.
    eps_0: float = 1.
    features: int = 64
    init_sigma: float = 1.
    modes: int = 20
    mem_len: int = 800
    n_layers: int = 3
    model_type: str = 'ng'
    ic_weight: float = 10.
    substeps: int = 5
    dtype: typing.Any = jnp.float_


class DieletricEmbedding(linen.Module):
    eps_0: float = 1.

    def __call__(self, r, dielectric_fn):
        if isinstance(dielectric_fn, DielectricVacuum):
            x = jnp.array([[1., 0.]])
        else:
            x = jnp.array([[0., 1.]])

        x = jnp.tile(x, [len(r), 1])

        eps_0 = self.eps_0
        eps_r = dielectric_fn(r)[..., 0:1]
        eps_scale = dielectric_fn.eps_max - eps_0

        if isinstance(dielectric_fn, DielectricVacuum):
            eps_ind = jnp.zeros_like(eps_r)
        else:
            eps_ind = (eps_r - eps_0) / eps_scale

        x = jnp.concatenate([x, eps_ind], axis=-1)
        return x


class ENet(linen.Module):
    config: MaxwellPropagatorModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        # K = KNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # B = BNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # W = WNet(self.config, 4)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        #
        # x_mu = jnp.concatenate([t, r], -1)
        # A_mu = W * jnp.exp(1j * (jnp.sum(K * x_mu[:, None], axis=-1, keepdims=True) + B))
        # A_mu = jnp.mean(A_mu, 1)

        features = self.config.features
        n_layers = self.config.n_layers
        c = self.config.c

        mat = DieletricEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        loc = jnp.asarray([light_source.loc])
        R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))
        # x = jnp.concatenate([c * t, R, mat], -1)
        # theta = jnp.angle(r[..., 0:1] / R + 1j * r[..., 1:2] / R)
        x = jnp.concatenate([c * t, r, R, mat], -1)

        psi = SIREN(features, n_layers=n_layers, omega0=1., out_dim=1)(x)

        x = jnp.concatenate([r / (R + 1e-6), mat], -1)
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)

        x0 = x
        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x) + x0

        x0 = x
        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x) + x0

        x0 = x
        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        w = linen.Dense(3)(x)

        E = jnp.concatenate([w * psi], -1)

        return E


class MaxwellPropagatorModel(linen.Module):
    config: MaxwellPropagatorModelConfig

    def setup(self):
        # self.phi_net = PhiNet(self.config)
        # self.A_net = ANet(self.config)
        self.E1_net = ENet(self.config)

    def get_observables(self, h, r, t, light_source, dielectric_fn):
        E = self(h, r, t, light_source, dielectric_fn)

        def grad_E1_op(_r, _t):
            E1_closure = lambda some_r, some_t: self.E1_net(h, some_r, some_t, light_source, dielectric_fn)
            grad_E1 = jax.vmap(jax.jacfwd(E1_closure))(_r[:, None], _t[:, None]).reshape(_r.shape + (3,))
            return grad_E1

        grad_E1 = grad_E1_op(r, t)

        err_em = grad_E1 - self.get_GVE(h, r, t, light_source, dielectric_fn)

        obs = dict(
            pred=E,
            err_em=err_em,
        )
        return obs

    def get_GVE(self, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        eps_0 = self.config.eps_0
        omega = light_source.omega
        q2 = omega ** 2 / c ** 2 * eps_0

        def green_fn(r0, r1):
            R = jnp.sqrt(jnp.sum((r0 - r1) ** 2, axis=-1, keepdims=True))
            return jnp.exp(1j * light_source.k0 * R) / (4 * jnp.pi * R)

        def grad_green_fn(r0, r1):
            grad_g = jax.jacfwd(green_fn)(r0, r1)
            return grad_g.reshape(r0.shape)

        def grad_grad_green_fn(r0, r1):
            grad_grad_g = jax.vmap(jax.jacfwd(grad_green_fn))(r0[:, None], r1[:, None])
            return grad_grad_g.reshape(r0.shape + (3,))

        def V_fn(r1):
            eps_r = dielectric_fn(r1)
            return omega ** 2 / c ** 2 * (eps_0 - eps_r)

        r = r[:, None]
        r_d = r[None, :]
        G = green_fn(r, r_d) + grad_grad_green_fn(r, r_d) / q2
        V = V_fn(r_d)[..., None]
        E0 = light_source.get_fields(r, t)
        E1 = self.E1_net(h, r, t, light_source, dielectric_fn)
        E = E0 + E1
        fields = G @ (V * E)

        return fields

    def __call__(self, h, r, t, light_source, dielectric_fn):
        # phi = self.phi_net(h, r, t, light_source, dielectric_fn)
        # A = self.A_net(h, r, t, light_source, dielectric_fn)
        # return phi, A
        E0 = light_source.get_fields(r, t)
        E1 = self.E1_net(h, r, t, light_source, dielectric_fn)
        E = E0 + E1
        return E


def create_maxwell_propagator_model(config: MaxwellPropagatorModelConfig):
    model = MaxwellPropagatorModel(config)

    def init():
        return

    def eval_step():
        return

    def sample_step():
        return

    def train_step():
        return

    return init, eval_step, sample_step, train_step
