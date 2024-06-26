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


class MaxwellPotentialModelConfig(struct.PyTreeNode):
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
    mem_len: int = 400
    n_layers: int = 3
    model_type: str = 'ng'
    envelope: str = 'gaussian'
    ic_weight: float = 10.
    gauge_choice: str = 'lorentz'
    substeps: int = 5
    dtype: typing.Any = jnp.float_
    model_type: str = 'SIREN'


class SpaceEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig
    embed_type: str = 'lattice'

    @linen.compact
    def __call__(self, r, light_source):
        features = self.config.features
        # r_dim = r.shape[-1]

        if self.embed_type == 'gaussian':
            W_r = self.param('W_r', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype),
                             (3, features))
            W_r = jax.lax.stop_gradient(W_r)
            W_r = W_r * light_source.k0
            r_emb = jnp.concatenate([jnp.sin(r @ W_r), jnp.cos(r @ W_r)], -1)
        elif self.embed_type == 'lattice':
            xy = r[..., :2]
            W_x = 2 ** jnp.linspace(-8, 0, 16)
            W_r = jnp.stack(jnp.meshgrid(*([W_x] * 2)), 0).reshape(2, -1)
            W_r = W_r * light_source.k0
            r_emb = jnp.concatenate([jnp.sin(xy @ W_r), jnp.cos(xy @ W_r)], -1)
        else:
            raise ValueError

        # loc = jnp.asarray([light_source.loc])
        # R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))
        # r_emb = jnp.concatenate([r_emb, r / (R + 1e-6)], -1)

        r_emb = linen.silu(linen.Dense(features)(r_emb))
        r_emb = linen.Dense(features)(r_emb)
        return r_emb


class TimeEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, t, light_source):
        features = self.config.features

        # W_t = self.param('W_t', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype), (1, features))
        # W_t = jax.lax.stop_gradient(W_t)
        # W_t = W_t * self.config.omega
        # t_emb = jnp.concatenate([jnp.sin(t @ W_t), jnp.cos(t @ W_t)], -1)

        W_t = 2 ** jnp.linspace(-8, 0, features)
        W_t = W_t.reshape(1, features) * light_source.omega
        t_emb = jnp.concatenate([jnp.sin(t @ W_t), jnp.cos(t @ W_t)], -1)

        t_emb = linen.silu(linen.Dense(features)(t_emb))
        t_emb = linen.Dense(features)(t_emb)

        return t_emb


class MaterialEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig

    def __call__(self, r, dielectric_fn):
        if isinstance(dielectric_fn, DielectricVacuum):
            x = jnp.array([[1., 0.]])
        else:
            x = jnp.array([[0., 1.]])

        x = jnp.tile(x, [len(r), 1])

        eps_0 = self.config.eps_0
        eps_r = dielectric_fn(r)[..., 0:1]
        eps_scale = dielectric_fn.eps_max - eps_0

        if isinstance(dielectric_fn, DielectricVacuum):
            eps_ind = jnp.zeros_like(eps_r)
        else:
            eps_ind = (eps_r - eps_0) / eps_scale

        x = jnp.concatenate([x, eps_ind], axis=-1)
        return x


class KNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        r_emb = SpaceEmbedding(self.config)(jax.lax.stop_gradient(r), light_source)
        t_emb = TimeEmbedding(self.config)(jax.lax.stop_gradient(t), light_source)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features, 'dnb')(h, x)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features * 2)(x)

        x = linen.Dense(modes * 4)(x)
        K0 = jnp.asarray([light_source.omega, light_source.k0, light_source.k0, light_source.k0])
        K = x.reshape(-1, modes, 4) * K0

        # x = linen.Dense(modes * 3)(x)
        # K = x.reshape(-1, modes, 3) * light_source.k0
        # c = self.config.c / jnp.sqrt(dielectric_fn(r)[..., 0:1])
        # K = jnp.concatenate([jnp.sqrt(jnp.sum(K ** 2, axis=-1, keepdims=True)) * c[:, None], K], -1)

        return K


class BNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        r_emb = SpaceEmbedding(self.config)(jax.lax.stop_gradient(r), light_source)
        t_emb = TimeEmbedding(self.config)(jax.lax.stop_gradient(t), light_source)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features, 'dnb')(h, x)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features * 2)(x)
        x = linen.Dense(modes)(x)

        B = x.reshape(-1, modes, 1) * jnp.pi
        return B


class PNet(linen.Module):
    config: MaxwellPotentialModelConfig
    out_dim: int

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        r_emb = SpaceEmbedding(self.config)(r, light_source)
        t_emb = TimeEmbedding(self.config)(t, light_source)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features)(h, x)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features * 2)(x)
        x = linen.Dense(modes * self.out_dim)(x)

        p = x.reshape(-1, modes, self.out_dim) / features
        # loc = jnp.asarray([light_source.loc])
        # R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))[:, None]
        # W = x.reshape(-1, modes, self.out_dim) / R
        return p


class PhiNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        features = self.config.features
        modes = self.config.modes

        k_mu = KNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        b = BNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # p = PNet(self.config, 1)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # x_mu = jnp.concatenate([c * t, r], -1)
        # psi = p * jnp.exp(1j * (jnp.sum(k_mu * x_mu[:, None], axis=-1, keepdims=True) + b))
        x_mu = jnp.concatenate([c * t, r], -1)
        psi = jnp.exp(1j * (jnp.sum(k_mu * x_mu[:, None], axis=-1, keepdims=True) + b))

        mat = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        loc = jnp.asarray([light_source.loc])
        R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))

        x = jnp.concatenate([r / (R + 1e-6), mat], -1)
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        w = linen.Dense(self.config.modes)(x)
        w = w[:, :, None]
        phi = w * psi.real

        # phi = jnp.mean(phi, 1)
        # a = self.param('a', linen.initializers.uniform(2 / modes), (modes, 1))
        # phi = jnp.sum(a * phi, 1)
        x = phi.transpose(0, 2, 1)
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        phi = linen.Dense(1)(x)
        phi = phi.squeeze(-1)

        return phi


class ANet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        features = self.config.features
        modes = self.config.modes

        k_mu = KNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        b = BNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # p = PNet(self.config, 3)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        # x_mu = jnp.concatenate([c * t, r], -1)
        # psi = p * jnp.exp(1j * (jnp.sum(k_mu * x_mu[:, None], axis=-1, keepdims=True) + b))
        x_mu = jnp.concatenate([c * t, r], -1)
        psi = jnp.exp(1j * (jnp.sum(k_mu * x_mu[:, None], axis=-1, keepdims=True) + b))

        mat = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        loc = jnp.asarray([light_source.loc])
        R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))

        x = jnp.concatenate([r / (R + 1e-6), mat], -1)
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        w = linen.Dense(self.config.modes * 3)(x)
        w = w.reshape(-1, self.config.modes, 3)
        A = w * psi.real

        # A = jnp.mean(A, 1)
        # a = self.param('a', linen.initializers.uniform(2 / modes), (modes, 1))
        # A = jnp.sum(a * A, 1)
        x = A.transpose(0, 2, 1)
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        A = linen.Dense(1)(x)
        A = A.squeeze(-1)
        return A


class AmuNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        if self.config.model_type == 'SIREN':
            features = self.config.features
            modes = self.config.modes
            n_layers = self.config.n_layers
            c = self.config.c

            mat = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
            loc = jnp.asarray([light_source.loc])
            R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))
            # theta = jnp.angle(r[..., 0:1] / R + 1j * r[..., 1:2] / R)

            x = jnp.concatenate([c * t, R, mat], -1)
            phi = SIREN(features, n_layers=n_layers, omega0=1., out_dim=1)(x)
            A = SIREN(features, n_layers=n_layers, omega0=1., out_dim=1)(x)

            x = jnp.concatenate([r / (R + 1e-6), mat], -1)
            x = linen.silu(linen.Dense(features)(x))
            x = linen.Dense(features)(x)

            for _ in range(n_layers):
                x0 = x
                x = linen.silu(linen.Dense(features)(x0))
                x = linen.Dense(features)(x) + x0

            x = linen.silu(linen.Dense(features * 2)(x))
            x = linen.Dense(features)(x)
            p = linen.Dense(4)(x)
            A_mu = jnp.concatenate([p[..., 0:1] * phi, p[..., 1:4] * A], -1)
        elif self.config.model_type == 'NG':
            modes = self.config.modes
            phi = PhiNet(self.config)(h, r, t, light_source, dielectric_fn)
            A = ANet(self.config)(h, r, t, light_source, dielectric_fn)
            A_mu = jnp.concatenate([phi, A], -1)

            a = self.param('a', linen.initializers.uniform(2 / modes), (modes, 1))
            A_mu = jnp.sum(a * A_mu, 1)
        else:
            raise NotImplementedError

        return A_mu


class MaxwellPotentialModel(linen.Module):
    config: MaxwellPotentialModelConfig

    # flow: FlowTransform
    # prior: distrax.Distribution

    def init_state(self, rng):
        # W_h = jax.random.normal(rng, (self.config.mem_len, self.config.features // 2))
        # h = 2 * jnp.pi * W_h
        # h_i = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)

        W_h = jax.random.normal(rng, (self.config.mem_len, self.config.features))
        # g_h = jnp.linspace(-1., 1., self.config.mem_len)[:, None]
        # h_i = jnp.exp(g_h) * W_h
        h_i = jnp.sqrt(1 / 2) * W_h
        # h_i = jnp.sqrt(1 / self.config.features) * W_h
        # W_h = jax.random.uniform(rng, (self.config.mem_len, self.config.features))
        # h_i = 2 * (W_h - 0.5)

        # h_i = jnp.linspace(-1., 1., self.config.mem_len)[:, None]
        return h_i

    def setup(self):
        self.phi_net = PhiNet(self.config)
        self.A_net = ANet(self.config)
        # self.A_mu_net = AmuNet(self.config)

    def get_phi(self, h, r, t, light_source, dielectric_fn):
        return self.phi_net(h, r, t, light_source, dielectric_fn)
        # return self.A_mu_net(h, r, t, light_source, dielectric_fn)[..., 0:1]

    def get_A(self, h, r, t, light_source, dielectric_fn):
        return self.A_net(h, r, t, light_source, dielectric_fn)
        # return self.A_mu_net(h, r, t, light_source, dielectric_fn)[..., 1:4]

    def get_observables(self, rng, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        eps_0 = self.config.eps_0

        pred = self.get_fields(h, r, t, light_source, dielectric_fn)

        def grad_phi_op(_r, _t):
            phi_closure = lambda some_r, some_t: self.get_phi(h, some_r, some_t, light_source, dielectric_fn)
            grad_phi = jax.vmap(jax.jacfwd(phi_closure))(_r[:, None], _t[:, None]).reshape(_r.shape)
            return grad_phi

        def grad_A_op(_r, _t):
            A_closure = lambda some_r, some_t: self.get_A(h, some_r, some_t, light_source, dielectric_fn)
            grad_A = jax.vmap(jax.jacfwd(A_closure))(_r[:, None], _t[:, None]).reshape(_r.shape + (3,))
            return grad_A

        def div_A_op(_r, _t):
            grad_A = grad_A_op(_r, _t)
            return jnp.trace(grad_A, axis1=-2, axis2=-1).reshape(_t.shape)

        def grad_div_A_op(_r, _t):
            grad_div_A = jax.vmap(jax.jacfwd(div_A_op))(_r[:, None], _t[:, None]).reshape(_r.shape)
            return grad_div_A

        def lap_phi_op(_r, _t):
            lap_phi = jax.vmap(jax.jacfwd(grad_phi_op))(_r[:, None], _t[:, None]).reshape(_r.shape + (3,))
            return jnp.trace(lap_phi, axis1=-2, axis2=-1).reshape(_t.shape)

        def lap_A_op(_r, _t):
            lap_A = jax.vmap(jax.jacfwd(grad_A_op))(_r[:, None], _t[:, None]).reshape(_r.shape + (3, 3))
            return jnp.trace(lap_A, axis1=-2, axis2=-1).reshape(_r.shape)

        def dot_phi_op(_r, _t):
            _, dot_phi = jax.jvp(lambda some_t: self.get_phi(h, _r, some_t, light_source, dielectric_fn),
                                 [_t], [jnp.ones_like(_t)])
            return dot_phi

        def dot_grad_phi_op(_r, _t):
            _, dot_grad_phi = jax.jvp(lambda some_t: grad_phi_op(_r, some_t),
                                      [_t], [jnp.ones_like(_t)])
            return dot_grad_phi

        def ddot_phi_op(_r, _t):
            _, ddot_phi = jax.jvp(lambda some_t: dot_phi_op(_r, some_t),
                                  [_t], [jnp.ones_like(_t)])
            return ddot_phi

        def dot_A_op(_r, _t):
            _, dot_A = jax.jvp(lambda some_t: self.get_A(h, _r, some_t, light_source, dielectric_fn),
                               [_t], [jnp.ones_like(_t)])
            return dot_A

        def dot_div_A_op(_r, _t):
            _, dot_div_A = jax.jvp(lambda some_t: div_A_op(_r, some_t),
                                   [_t], [jnp.ones_like(_t)])
            return dot_div_A

        def ddot_A_op(_r, _t):
            _, ddot_A = jax.jvp(lambda some_t: dot_A_op(_r, some_t),
                                [_t], [jnp.ones_like(_t)])
            return ddot_A

        eps_r = dielectric_fn(r)[..., 0:1]
        rho = light_source.get_charge(r, t)
        j = light_source.get_current(r, t)

        if self.config.gauge_choice == 'lorentz':
            res1 = lap_phi_op(r, t) - eps_r / (c ** 2) * ddot_phi_op(r, t) + rho / eps_0
            res2 = lap_A_op(r, t) - eps_r / (c ** 2) * ddot_A_op(r, t) + j / eps_0 / c ** 2
            res3 = div_A_op(r, t) + eps_r / (c ** 2) * dot_phi_op(r, t)
            err_em = jnp.sum(jnp.real(res1.conj() * res1))
            err_em += jnp.sum(jnp.real(res2.conj() * res2))
            err_em += jnp.sum(jnp.real(res3.conj() * res3))
        else:
            res1 = lap_phi_op(r, t) + dot_div_A_op(r, t) + rho / eps_0
            res2 = (c ** 2) / eps_r * lap_A_op(r, t) - (c ** 2) / eps_r * grad_div_A_op(r, t) - dot_grad_phi_op(r, t) - ddot_A_op(r, t) + j / eps_0
            err_em = jnp.sum(jnp.real(res1.conj() * res1))
            err_em += jnp.sum(jnp.real(res2.conj() * res2))

        obs = dict(
            pred=pred,
            err_em=err_em,
            logq=0.
        )
        return obs

    def __call__(self, h, r, t, light_source, dielectric_fn):
        phi = self.phi_net(h, r, t, light_source, dielectric_fn)
        A = self.A_net(h, r, t, light_source, dielectric_fn)
        return jnp.concatenate([phi, A], -1)
        # A_mu = self.A_mu_net(h, r, t, light_source, dielectric_fn)
        # return A_mu

    def get_fields(self, h, r, t, light_source, dielectric_fn):
        phi = self.get_phi(h, r, t, light_source, dielectric_fn)
        A = self.A_net(h, r, t, light_source, dielectric_fn)
        phi_closure = lambda _r, _t: self.get_phi(h, _r, _t, light_source, dielectric_fn)
        grad_phi = jax.vmap(jax.jacfwd(phi_closure))(r[:, None], t[:, None]).reshape(r.shape)
        _, dot_A = jax.jvp(lambda _t: self.get_A(h, r, _t, light_source, dielectric_fn),
                           [t], [jnp.ones_like(t)])
        dot_A = dot_A.reshape(r.shape)
        E = -grad_phi - dot_A
        return dict(E=E, phi=phi, A=A)

    def update(self, h, r, t, v, light_source, dielectric_fn):
        dt = self.config.dt

        r_next = r + v * dt
        t_next = t + dt
        v_next = v

        return h, r_next, t_next, v_next


def create_maxwell_potential_model(config: MaxwellPotentialModelConfig):
    model = MaxwellPotentialModel(config)

    def init(rng, ic, light_source, dielectric_fn):
        keys = jax.random.split(rng, 9)

        r, t, v = ic

        h = model.init_state(keys[0])

        vars = model.init(keys[1], keys[2], h, r, t, light_source, dielectric_fn, method=model.get_observables)
        params = vars['params']
        return model, params

    def sample_step(params, rng, h, r, t, v, light_source, dielectric_fn):
        return model.apply({'params': params}, h, r, t, v, light_source, dielectric_fn, method=model.update)

    def sample_train(params, rng, h_i, r_i, t_i, v_i, light_source, dielectric_fn, lamb=1.0):
        keys = jax.random.split(rng, config.sample_length)

        def update_step(carry, key):
            _h, _r, _t, _v = carry
            _h_next, _r_next, _t_next, _v_next = model.apply({'params': params}, _h, _r, _t, _v, light_source,
                                                             dielectric_fn, method=model.update)
            return (_h_next, _r_next, _t_next, _v_next), (_h_next, _r_next, _t_next, _v_next)

        _, (h_traj, r_traj, t_traj, v_traj) = jax.lax.scan(update_step, (h_i, r_i, t_i, v_i), keys)

        return h_traj, r_traj, t_traj, v_traj

    def get_observables(params, rngs, h_batch, r_batch, t_batch, light_source, dielectric_fn):
        def apply_fn(key, h, r, t):
            return model.apply({'params': params}, key, h, r, t, light_source, dielectric_fn,
                               method=model.get_observables)

        obs = jax.vmap(apply_fn, (0, 0, 0, 0))(rngs, h_batch, r_batch, t_batch)
        return obs

    def get_pred(params, h_batch, r_batch, t_batch, light_source, dielectric_fn):
        def apply_fn(h, r, t):
            return model.apply({'params': params}, h, r, t, light_source, dielectric_fn, method=model.get_fields)

        pred = jax.vmap(apply_fn, (0, 0, 0))(h_batch, r_batch, t_batch)
        return pred

    def loss_fn(params, rng, ic, light_source, dielectric_fn, lamb=1.0):
        r_i, t_i, v_i = ic

        rng, key = jax.random.split(rng)
        h_i = model.init_state(key)
        # h_traj, r_traj, t_traj, v_traj = sample_train(params, key, h_i, r_i, t_i, v_i, light_source, dielectric_fn, lamb)

        # rng, *keys = jax.random.split(rng, h_traj.shape[0] + 1)
        # obs_traj = get_observables(params, jnp.asarray(keys), h_traj, r_traj, t_traj, light_source, dielectric_fn)
        # err_pde = obs_traj['err_em']
        # err_pde = jnp.sum(err_pde)

        if isinstance(dielectric_fn, DielectricVacuum):
            obs = model.apply({'params': params}, key, h_i, r_i, t_i, light_source, dielectric_fn,
                              method=model.get_observables)
            pred = obs['pred']
            err_pde = jnp.sum(obs['err_em'])
            # err_pde = 0.

            imp_weights = 1.
            # loc = jnp.asarray([light_source.loc])
            # imp_weights = jnp.sum((r_vac - loc) ** 2, axis=-1, keepdims=True)
            # phi_pred, A_pred = model.apply({'params': params}, h_i, r_i, t_i, light_source, dielectric_fn)
            # E_pred = model.apply({'params': params}, h_i, r_vac, t_vac, light_source, dielectric_fn, method=model.get_fields)
            phi_target, A_target = light_source.get_potentials(r_i, t_i)
            E_target = light_source.get_fields(r_i, t_i)
            err_sup = jnp.sum(jnp.abs(jnp.real(pred['E']) - jnp.real(E_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['phi']) - jnp.real(phi_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['A']) - jnp.real(A_target)) ** 2 * imp_weights)
            # err_sup = 0.
        else:
            obs = model.apply({'params': params}, key, h_i, r_i, t_i, light_source, dielectric_fn,
                              method=model.get_observables)
            err_pde = jnp.sum(obs['err_em'])

            source_loc = jnp.asarray([light_source.loc])
            n_near = 1000
            r_near = jax.random.normal(rng, (n_near, 3)) * 2 * jnp.pi / light_source.k0 * 0.1 + source_loc
            t_near = jnp.zeros((n_near, 1)) + config.t_domain[0]
            pred = model.apply({'params': params}, h_i, r_near, t_near, light_source, dielectric_fn,
                               method=model.get_fields)
            phi_target, A_target = light_source.get_potentials(r_near, t_near)
            E_target = light_source.get_fields(r_near, t_near)
            imp_weights = 1.
            err_sup = jnp.sum(jnp.abs(jnp.real(pred['E']) - jnp.real(E_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['phi']) - jnp.real(phi_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['A']) - jnp.real(A_target)) ** 2 * imp_weights)

            # obs = model.apply({'params': params}, key, h_i, r_i, t_i, light_source, DielectricVacuum(), method=model.get_observables)
            # err_pde += jnp.sum(obs['err_em'])

            imp_weights = 1.
            # loc = jnp.asarray([light_source.loc])
            # imp_weights = jnp.sum((r_vac - loc) ** 2, axis=-1, keepdims=True)
            pred = model.apply({'params': params}, h_i, r_i, t_i, light_source, DielectricVacuum(),
                               method=model.get_fields)
            phi_target, A_target = light_source.get_potentials(r_i, t_i)
            E_target = light_source.get_fields(r_i, t_i)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['E']) - jnp.real(E_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['phi']) - jnp.real(phi_target)) ** 2 * imp_weights)
            err_sup += jnp.sum(jnp.abs(jnp.real(pred['A']) - jnp.real(A_target)) ** 2 * imp_weights)
            # err_sup = 0.

        loss = err_pde + 100 * err_sup
        stats = dict(loss=loss, err_pde=err_pde, err_sup=err_sup)
        return loss, stats

    def eval_step(params, rng, h, r, t, light_source, dielectric_fn, lamb=1.0):
        return model.apply({'params': params}, h, r, t, light_source, dielectric_fn, method=model.get_fields)

    def train_step(state: TrainState, rng: chex.PRNGKey, ic: typing.Tuple, light_source,
                   dielectric_fn: GenericMaterial, lamb: float):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, stats), grads = grad_fn(state.params, rng, ic, light_source, dielectric_fn, lamb)
        # grads = jax.tree_map(lambda g: jnp.clip(g, -1., 1.), grads)
        state, updates = state.apply_gradients(grads=grads)
        return state, stats

    return init, eval_step, sample_step, train_step
