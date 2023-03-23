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
    mem_len: int = 800
    n_layers: int = 3
    model_type: str = 'ng'
    envelope: str = 'gaussian'
    ic_weight: float = 10.
    eom: str = 'bohm'
    substeps: int = 5
    dtype: typing.Any = jnp.float_
    log_domain: bool = True
    use_ode: bool = False


class SpaceEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig
    embed_type: str = 'lattice'

    @linen.compact
    def __call__(self, r, light_source):
        features = self.config.features
        # r_dim = r.shape[-1]

        if self.embed_type == 'gaussian':
            W_r = self.param('W_r', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype), (3, features))
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

        x = MaterialEmbedding(self.config)(r, dielectric_fn)
        r_emb = SpaceEmbedding(self.config)(r, light_source)
        t_emb = TimeEmbedding(self.config)(t, light_source)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features)(h, x)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features * 2)(x)

        # x = linen.Dense(modes * 4)(x)
        # K0 = jnp.asarray([light_source.omega, light_source.k0, light_source.k0, light_source.k0])
        # K = x.reshape(-1, modes, 4) * K0

        x = linen.Dense(modes * 3)(x)
        K = x.reshape(-1, modes, 3) * light_source.k0
        c = self.config.c / jnp.sqrt(dielectric_fn(r)[..., 0:1])
        K = jnp.concatenate([jnp.sqrt(jnp.sum(K ** 2, axis=-1, keepdims=True)) * c[:, None], K], -1)

        return K


class BNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(r, dielectric_fn)
        r_emb = SpaceEmbedding(self.config)(r, light_source)
        t_emb = TimeEmbedding(self.config)(t, light_source)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features)(h, x)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features * 2)(x)
        x = linen.Dense(modes)(x)

        B = x.reshape(-1, modes, 1) * jnp.pi
        return B


class WNet(linen.Module):
    config: MaxwellPotentialModelConfig
    out_dim: int

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(r, dielectric_fn)
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

        W = x.reshape(-1, modes, self.out_dim) / jnp.sqrt(features)
        # loc = jnp.asarray([light_source.loc])
        # R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))[:, None]
        # W = x.reshape(-1, modes, self.out_dim) / R
        return W


class PhiNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        K = KNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        B = BNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        W = WNet(self.config, 1)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)

        x_mu = jnp.concatenate([t, r], -1)
        phi = W * jnp.exp(1j * (jnp.sum(K * x_mu[:, None], axis=-1, keepdims=True) + B))
        phi = jnp.mean(phi, 1)
        return phi


class ANet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        K = KNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        B = BNet(self.config)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)
        W = WNet(self.config, 3)(h, jax.lax.stop_gradient(r), jax.lax.stop_gradient(t), light_source, dielectric_fn)

        x_mu = jnp.concatenate([t, r], -1)
        A = W * jnp.exp(1j * (jnp.sum(K * x_mu[:, None], axis=-1, keepdims=True) + B))
        A = jnp.mean(A, 1)
        return A


class AmuNet(linen.Module):
    config: MaxwellPotentialModelConfig

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

        x = MaterialEmbedding(self.config)(jax.lax.stop_gradient(r), dielectric_fn)
        # loc = jnp.asarray([light_source.loc])
        # R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))
        x = jnp.concatenate([c * t, r, x], -1)

        A_mu = SIREN(features, n_layers=n_layers, omega0=2 * jnp.pi,  out_dim=4)(x)
        # phi = SIREN(features, n_layers=n_layers, omega0=2 * jnp.pi, out_dim=1)(x)
        # A = SIREN(features, n_layers=n_layers, omega0=2 * jnp.pi, out_dim=3)(x)
        # A_mu = jnp.concatenate([phi, A], -1)

        # loc = jnp.asarray([light_source.loc])
        # R = jnp.sqrt(jnp.sum((r - loc) ** 2, axis=-1, keepdims=True))
        # A_mu /= R

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
        # self.phi_net = PhiNet(self.config)
        # self.A_net = ANet(self.config)
        self.A_mu_net = AmuNet(self.config)

    def get_phi(self, h, r, t, light_source, dielectric_fn):
        return self.A_mu_net(h, r, t, light_source, dielectric_fn)[..., 0:1]

    def get_A(self, h, r, t, light_source, dielectric_fn):
        return self.A_mu_net(h, r, t, light_source, dielectric_fn)[..., 1:4]

    def get_observables(self, rng, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        eps_0 = self.config.eps_0

        E_field = self.get_fields(h, r, t, light_source, dielectric_fn)

        def grad_phi_op(_r, _t):
            phi_closure = lambda some_r, _t: self.get_phi(h, some_r, _t, light_source, dielectric_fn)
            grad_phi = jax.vmap(jax.jacfwd(phi_closure))(_r[:, None], _t[:, None])
            return grad_phi.reshape(_r.shape)

        def grad_A_op(_r, _t):
            _, grad_A = jax.jvp(lambda some_r: self.get_A(h, some_r, _t, light_source, dielectric_fn),
                                [_r], [jnp.ones_like(_r)])
            return grad_A

        def div_A_op(_r, _t):
            grad_A = grad_A_op(_r, _t)
            return grad_A.sum(axis=-1, keepdims=True)

        def lap_phi_op(_r, _t):
            _, lap_phi = jax.jvp(lambda some_r: grad_phi_op(some_r, _t),
                                 [_r], [jnp.ones_like(_r)])
            return lap_phi.sum(axis=-1, keepdims=True)

        def lap_A_op(_r, _t):
            _, lap_A = jax.jvp(lambda some_r: grad_A_op(some_r, _t),
                               [_r], [jnp.ones_like(_r)])
            return lap_A.sum(axis=-1, keepdims=True)

        def dot_phi_op(_r, _t):
            _, dot_phi = jax.jvp(lambda some_t: self.get_phi(h, _r, some_t, light_source, dielectric_fn),
                                 [_t], [jnp.ones_like(_t)])
            return dot_phi

        def ddot_phi_op(_r, _t):
            _, ddot_phi = jax.jvp(lambda some_t: dot_phi_op(_r, some_t),
                                  [_t], [jnp.ones_like(_t)])
            return ddot_phi

        def dot_A_op(_r, _t):
            _, dot_A = jax.jvp(lambda some_t: self.get_A(h, _r, some_t, light_source, dielectric_fn),
                               [_t], [jnp.ones_like(_t)])
            return dot_A

        def ddot_A_op(_r, _t):
            _, ddot_A = jax.jvp(lambda some_t: dot_A_op(_r, some_t),
                                [_t], [jnp.ones_like(_t)])
            return ddot_A

        eps_r = dielectric_fn(r)[..., 0:1]
        rho = light_source.get_charge(r, t)
        j = light_source.get_current(r, t)
        loss_phi = lap_phi_op(r, t) - eps_r / (c ** 2) * ddot_phi_op(r, t) + rho / eps_0
        loss_A = lap_A_op(r, t) - eps_r / (c ** 2) * ddot_A_op(r, t) + j / eps_0 / c ** 2
        loss_gauge = div_A_op(r, t) + eps_r / (c ** 2) * dot_phi_op(r, t)

        err_em = jnp.sum(jnp.real(loss_phi.conj() * loss_phi))
        err_em += jnp.sum(jnp.real(loss_A.conj() * loss_A))
        err_em += jnp.sum(jnp.real(loss_gauge.conj() * loss_gauge))

        obs = dict(
            E_field=E_field,
            err_em=err_em,
            logq=0.
        )
        return obs

    def __call__(self, h, r, t, light_source, dielectric_fn):
        # phi = self.phi_net(h, r, t, light_source, dielectric_fn)
        # A = self.A_net(h, r, t, light_source, dielectric_fn)
        # return phi, A
        A_mu = self.A_mu_net(h, r, t, light_source, dielectric_fn)
        return A_mu[..., 0:1], A_mu[..., 1:4]

    def get_fields(self, h, r, t, light_source, dielectric_fn):
        grad_phi = jax.vmap(jax.jacfwd(lambda _r, _t: self.get_phi(h, _r, _t, light_source, dielectric_fn)))(r[:, None], t[:, None])
        grad_phi = grad_phi.reshape(r.shape)
        _, dot_A = jax.jvp(lambda _t: self.get_A(h, r, _t, light_source, dielectric_fn),
                           [t], [jnp.ones_like(t)])
        dot_A = dot_A.reshape(r.shape)
        E = -grad_phi - dot_A
        return E

    def update(self, h, r, t, v, light_source, dielectric_fn):
        dt = self.config.dt

        r_next = r + v * dt
        t_next = t + dt
        v_next = v

        return h, r_next, t_next, v_next


def create(config: MaxwellPotentialModelConfig):

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
            _h_next, _r_next, _t_next, _v_next = model.apply({'params': params}, _h, _r, _t, _v, light_source, dielectric_fn, method=model.update)
            return (_h_next, _r_next, _t_next, _v_next), (_h_next, _r_next, _t_next, _v_next)

        _, (h_traj, r_traj, t_traj, v_traj) = jax.lax.scan(update_step, (h_i, r_i, t_i, v_i), keys)

        return h_traj, r_traj, t_traj, v_traj

    def get_observables(params, rngs, h_batch, r_batch, t_batch, light_source, dielectric_fn):
        def apply_fn(key, h, r, t):
            return model.apply({'params': params}, key, h, r, t, light_source, dielectric_fn, method=model.get_observables)

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
        h_i = model.init_state(rng)
        # h_traj, r_traj, t_traj, v_traj = sample_train(params, key, h_i, r_i, t_i, v_i, light_source, dielectric_fn, lamb)

        # rng, *keys = jax.random.split(rng, h_traj.shape[0] + 1)
        # obs_traj = get_observables(params, jnp.asarray(keys), h_traj, r_traj, t_traj, light_source, dielectric_fn)
        # err_pde = obs_traj['err_em']
        # err_pde = jnp.sum(err_pde)

        rng, key = jax.random.split(rng)
        loc = jnp.asarray([light_source.loc])
        t_vac = jnp.zeros(t_i.shape) + config.t_domain[0]
        x_vac = jax.random.uniform(key, t_i.shape, minval=config.x_domain[0], maxval=config.x_domain[1])
        y_vac = jax.random.uniform(key, t_i.shape, minval=config.y_domain[0], maxval=config.y_domain[1])
        z_vac = jnp.zeros(t_i.shape)
        r_vac = jnp.concatenate([x_vac, y_vac, z_vac], -1)

        phi_pred, A_pred = model.apply({'params': params}, h_i, r_vac, t_vac, light_source, DielectricVacuum())
        # obs = model.apply({'params': params}, key, h_i, r_vac, t_vac, light_source, dielectric_fn, method=model.get_observables)
        # E_pred = obs['E_field']
        # err_pde = jnp.sum(obs['err_em'])
        E_pred = model.apply({'params': params}, h_i, r_vac, t_vac, light_source, DielectricVacuum(), method=model.get_fields)
        err_pde = 0.

        phi_target, A_target = light_source.get_potentials(r_vac, t_vac)
        E_target = light_source.get_fields(r_vac, t_vac)

        imp_weights = jnp.sum((r_vac - loc) ** 2, axis=-1, keepdims=True)
        # imp_weights = 1.
        err_sup = jnp.sum(jnp.abs(jnp.real(phi_pred) - jnp.real(phi_target)) ** 2 * imp_weights)
        err_sup += jnp.sum(jnp.abs(jnp.real(A_pred) - jnp.real(A_target)) ** 2 * imp_weights)
        err_sup += jnp.sum(jnp.abs(jnp.real(E_pred) - jnp.real(E_target)) ** 2 * imp_weights)

        loss = err_pde + err_sup
        stats = dict(loss=loss, err_pde=err_pde, err_sup=err_sup)
        return loss, stats

    def eval_step(params, rng, h_batch, r_batch, t_batch, light_source, dielectric_fn, lamb=1.0):
        preds = get_pred(params, h_batch, r_batch, t_batch, light_source, dielectric_fn)
        return preds

    def train_step(state: TrainState, rng: chex.PRNGKey, ic: typing.Tuple, light_source,
                   dielectric_fn: GenericMaterial, lamb: float):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, stats), grads = grad_fn(state.params, rng, ic, light_source, dielectric_fn, lamb)
        # grads = jax.tree_map(lambda g: jnp.clip(g, -1., 1.), grads)
        state, updates = state.apply_gradients(grads=grads)
        return state, stats

    return init, eval_step, sample_step, train_step
