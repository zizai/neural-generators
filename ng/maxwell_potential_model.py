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
    omega: float = 2 * math.pi / 1.5
    wavelength: float = 1.5
    E0_norm: float = 1.
    fs_l: float = 0.3
    features: int = 64
    init_sigma: float = 1.
    modes: int = 20
    mem_len: int = 800
    n_layers: int = 3
    envelope: str = 'gaussian'
    ic_weight: float = 10.
    eom: str = 'bohm'
    substeps: int = 5
    dtype: typing.Any = jnp.float_
    log_domain: bool = True
    use_ode: bool = False


class SpaceEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, r):
        features = self.config.features
        # r_dim = r.shape[-1]
        r_dim = 2
        r = r[..., :2]

        # W_r = self.param('W_r', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype), (r_dim, features))
        # W_r = jax.lax.stop_gradient(W_r)
        # W_r = W_r * 2 * jnp.pi / self.config.wavelength
        # r_emb = jnp.concatenate([jnp.sin(r @ W_r), jnp.cos(r @ W_r)], -1)

        W_x = 2 ** jnp.linspace(-8, 0, 16)
        W_r = jnp.stack(jnp.meshgrid(*([W_x] * r_dim)), 0).reshape(r_dim, -1)
        W_r = W_r * 2 * jnp.pi / self.config.wavelength
        r_emb = jnp.concatenate([jnp.sin(r @ W_r), jnp.cos(r @ W_r)], -1)

        # angles = jnp.arange(0, modes) / modes * 2 * jnp.pi
        # W_r = jnp.stack([jnp.cos(angles), jnp.sin(angles)], 0) * 2 * jnp.pi / self.config.wavelength
        # r_emb = jnp.concatenate([jnp.sin(r @ W_r), jnp.cos(r @ W_r)], -1)

        # W_r = 2 ** (jnp.arange(0, modes // 2) - modes / 4)
        # W_r = W_r * jnp.pi / self.config.wavelength
        # r_emb = jnp.concatenate([jnp.sin(r[..., None] * W_r), jnp.cos(r[..., None] * W_r)], -1).reshape(-1, modes * r_dim)

        r_emb = linen.silu(linen.Dense(features)(r_emb))
        r_emb = linen.Dense(features)(r_emb)
        return r_emb


class TimeEmbedding(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, t):
        features = self.config.features

        # W_t = self.param('W_t', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype), (1, features))
        # W_t = jax.lax.stop_gradient(W_t)
        # W_t = W_t * self.config.omega
        # t_emb = jnp.concatenate([jnp.sin(t @ W_t), jnp.cos(t @ W_t)], -1)

        W_t = 2 ** jnp.linspace(-8, 0, features)
        W_t = W_t.reshape(1, features) * self.config.omega
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
        eps_r = dielectric_fn(jax.lax.stop_gradient(r))[..., 0:1]
        eps_scale = dielectric_fn.eps_max - eps_0

        if isinstance(dielectric_fn, DielectricVacuum):
            eps_ind = jnp.zeros_like(eps_r)
        else:
            eps_ind = (eps_r - eps_0) / eps_scale

        x = jnp.concatenate([x, eps_ind], axis=-1)
        return x


class PhiNet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(r, dielectric_fn)

        r_emb = SpaceEmbedding(self.config)(r)
        t_emb = TimeEmbedding(self.config)(t)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features)(h, x)

        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)
        phi = linen.Dense(1)(x)
        return phi


class ANet(linen.Module):
    config: MaxwellPotentialModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        r_dim = r.shape[-1]

        x = MaterialEmbedding(self.config)(r, dielectric_fn)

        r_emb = SpaceEmbedding(self.config)(r)
        t_emb = TimeEmbedding(self.config)(t)
        x = jnp.concatenate([x, r_emb, t_emb], -1)

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            h, x = NeuralGeneratorLayer(features)(h, x)

        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x)
        A = linen.Dense(r_dim)(x)
        return A


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
        # self.e_net = SIREN(self.config)

    def get_observables(self, rng, h, r, t, light_source, dielectric_fn):
        c = self.config.c
        eps_0, wavelength = self.config.eps_0, self.config.wavelength

        # source_r, source_t = light_source.sample(rng, n_samples=1)
        # r_ls = light_source.mu
        # t_ls = t_l[:1]
        # r = jnp.concatenate([r, source_r], 0)
        # t = jnp.concatenate([t, source_t], 0)

        # r = r / au_const.nm * 1e-3
        # dL = 1 / au_const.nm * 1e-3
        # wavelength = wavelength / dL
        # omega =

        E_field = self.get_fields(h, r, t, light_source, dielectric_fn)
        # V = q * jnp.sum(r * E_field, axis=-1, keepdims=True)

        def grad_phi_op(_r, _t):
            phi_closure = lambda some_r, _t: self.phi_net(h, some_r, _t, light_source, dielectric_fn)
            grad_phi = jax.vmap(jax.jacfwd(phi_closure))(_r[:, None], _t[:, None])
            return grad_phi.reshape(_r.shape)

        def grad_A_op(_r, _t):
            _, grad_A = jax.jvp(lambda some_r: self.A_net(h, some_r, _t, light_source, dielectric_fn),
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
            _, dot_phi = jax.jvp(lambda some_t: self.phi_net(h, _r, some_t, light_source, dielectric_fn),
                                 [_t], [jnp.ones_like(_t)])
            return dot_phi

        def ddot_phi_op(_r, _t):
            _, ddot_phi = jax.jvp(lambda some_t: dot_phi_op(_r, some_t),
                                  [_t], [jnp.ones_like(_t)])
            return ddot_phi

        def dot_A_op(_r, _t):
            _, dot_A = jax.jvp(lambda some_t: self.A_net(h, _r, some_t, light_source, dielectric_fn),
                               [_t], [jnp.ones_like(_t)])
            return dot_A

        def ddot_A_op(_r, _t):
            _, ddot_A = jax.jvp(lambda some_t: dot_A_op(_r, some_t),
                                [_t], [jnp.ones_like(_t)])
            return ddot_A

        eps_r = dielectric_fn(r)
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
        phi = self.phi_net(h, r, t, light_source, dielectric_fn)
        A = self.A_net(h, r, t, light_source, dielectric_fn)
        return phi, A

    def get_fields(self, h, r, t, light_source, dielectric_fn):
        grad_phi = jax.vmap(jax.jacfwd(lambda _r, _t: self.phi_net(h, _r, _t, light_source, dielectric_fn)))(r[:, None], t[:, None])
        grad_phi = grad_phi.reshape(r.shape)
        _, dot_A = jax.jvp(lambda _t: self.A_net(h, r, _t, light_source, dielectric_fn),
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
        h_traj, r_traj, t_traj, v_traj = sample_train(params, key, h_i, r_i, t_i, v_i, light_source, dielectric_fn, lamb)
        # h_traj, r_traj, t_traj = data

        rng, *keys = jax.random.split(rng, h_traj.shape[0] + 1)
        obs_traj = get_observables(params, jnp.asarray(keys), h_traj, r_traj, t_traj, light_source, dielectric_fn)

        err_pde = obs_traj['err_em']
        err_pde = jnp.sum(err_pde)

        rng, key = jax.random.split(rng)
        loc = jnp.asarray([light_source.loc])
        t_vac = jnp.zeros(t_traj.shape[1:]) + config.t_domain[0]
        x_vac = jax.random.uniform(key, t_traj.shape[1:], minval=config.x_domain[0], maxval=config.x_domain[1])
        y_vac = jax.random.uniform(key, t_traj.shape[1:], minval=config.y_domain[0], maxval=config.y_domain[1])
        z_vac = jnp.zeros(t_traj.shape[1:])
        r_vac = jnp.concatenate([x_vac, y_vac, z_vac], -1)
        phi_pred, A_pred = model.apply({'params': params}, h_i, r_vac, t_vac, light_source, DielectricVacuum())
        phi_target, A_target = light_source.get_potentials(r_vac, t_vac)
        imp_weights = jnp.sum((r_vac - loc) ** 2, axis=-1, keepdims=True)
        err_sup = jnp.sum(jnp.abs(jnp.real(phi_pred) - jnp.real(phi_target)) ** 2 * imp_weights)
        err_sup += jnp.sum(jnp.abs(jnp.real(A_pred) - jnp.real(A_target)) ** 2 * imp_weights)

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
