import math
from functools import partial

import chex
import distrax
import jax
import jax.numpy as jnp
import typing
from flax import linen, struct

from ng import au_const
from ng.materials import GenericMaterial
from ng.ng_layer import NeuralGeneratorLayer
from ng.potentials import GenericPotential
from ng.train_state import TrainState


class MaxwellModelConfig(struct.PyTreeNode):
    t_i: float
    t_f: float
    dt: float
    sample_length: int
    hbar: float = au_const.hbar
    m: float = au_const.m_e
    c: float = au_const.c
    eps_0: float = au_const.eps_0
    q: float = au_const.q_e
    omega: float = 2 * math.pi * au_const.c / 800 / au_const.nm
    wavelength: float = 800 * au_const.nm
    E0_norm: float = 1.
    fs_l: float = 0.3
    r_dim: int = 2
    features: int = 64
    init_sigma: float = 1.
    modes: int = 20
    mem_len: int = 200
    n_layers: int = 3
    envelope: str = 'gaussian'
    ic_weight: float = 10.
    eom: str = 'bohm'
    substeps: int = 5
    dtype: typing.Any = jnp.float_
    log_domain: bool = True
    use_ode: bool = False


class SpaceEmbedding(linen.Module):
    config: MaxwellModelConfig

    @linen.compact
    def __call__(self, r):
        features = self.config.features
        r_dim = r.shape[-1]

        # W_r = self.param('W_r', jax.nn.initializers.normal(stddev=self.config.init_sigma, dtype=self.config.dtype), (r_dim, features))
        # W_r = jax.lax.stop_gradient(W_r)
        # W_r = W_r * 2 * jnp.pi / self.config.wavelength
        # r_emb = jnp.concatenate([jnp.sin(r @ W_r), jnp.cos(r @ W_r)], -1)

        W_x = 2 ** jnp.linspace(-8, 0, 32)
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
    config: MaxwellModelConfig

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


class ENet(linen.Module):
    config: MaxwellModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        omega = self.config.omega

        k = KNet(self.config)(h, r, t, light_source, dielectric_fn)
        # b = light_source.mu[:, None]
        shift = SNet(self.config)(h, r, t, light_source, dielectric_fn)

        w = WNet(self.config)(h, r, t, light_source, dielectric_fn)
        # w_real = WNet(self.config)(h, r, t, light_source, dielectric_fn)
        # w_imag = WNet(self.config)(h, r, t, light_source, dielectric_fn)

        # phi = jnp.exp(1j * (jnp.sum(k * (r[:, None] - light_source.loc[:, None]), axis=-1) + s - omega * t))
        v = self.config.c / jnp.sqrt(dielectric_fn(r))
        phi = jnp.exp(1j * (jnp.sum(k * r[:, None], axis=-1) - jnp.sum(k * v[:, None], axis=-1) * t + shift))
        phi = phi[:, None]
        E = jnp.mean(w * phi, -1)
        # E = jnp.sum((w_real + 1j * w_imag) * phi, -1)
        return E


class KNet(linen.Module):
    config: MaxwellModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        heads = int(features // 32)
        r_dim = r.shape[-1]

        r = jax.lax.stop_gradient(r)
        t = jax.lax.stop_gradient(t)

        eps_r = dielectric_fn(r)[..., 0:1]
        min_noise = jnp.finfo(self.config.dtype).resolution
        eps_min = eps_r.min()
        eps_scale = eps_r.max() - eps_r.min()
        x0 = jnp.where(eps_r.max() == eps_r.min(), jnp.zeros_like(eps_r), (eps_r - eps_min) / (eps_scale + min_noise))
        x0 = jnp.concatenate([x0, 1 - x0], axis=-1)

        # delta_r = jnp.alltrue(jnp.equal(r, light_source.loc), axis=-1, keepdims=True)
        # delta_r = jnp.exp(-jnp.sum((r - light_source.loc) ** 2, axis=-1, keepdims=True) / 2 / (self.config.wavelength / 10) ** 2)
        # x0 = jnp.concatenate([x0, delta_r], axis=-1)

        # r_norm = jnp.sqrt(jnp.sum((r - light_source.loc) ** 2, axis=-1, keepdims=True))
        # k_dir = jnp.clip((r - light_source.loc) / (r_norm + 1e-5), -1, 1)
        # x0 = jnp.concatenate([x0, k_dir], axis=-1)

        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x)
        # x = linen.LayerNorm()(x)

        r_emb = SpaceEmbedding(self.config)(r)
        t_emb = TimeEmbedding(self.config)(t)

        x = jnp.concatenate([x, r_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        x = jnp.concatenate([x, t_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x = NeuralGeneratorLayer(features)(x)

        x = linen.silu(linen.Dense(features)(x))
        k = linen.Dense(r_dim * modes)(x)
        k = k.reshape(-1, modes, r_dim)
        # k = k + k0[:, None]
        k = k * light_source.k0

        # x = linen.silu(linen.Dense(features)(x))
        # theta = linen.Dense(features // 2)(x).reshape(-1, features // 2) * jnp.pi
        # mat = jnp.array([
        #     [jnp.cos(theta), -jnp.sin(theta)],
        #     [jnp.sin(theta), jnp.cos(theta)]
        # ]).transpose(2, 3, 0, 1)
        # k = jnp.squeeze(mat @ k0[:, None, :, None], -1)

        # x = linen.silu(linen.Dense(features)(x))
        # params = linen.Dense(2)(x)
        # g, theta = params[..., 0:1], params[..., 1:2]
        # theta = theta * jnp.pi
        # s = jnp.exp(g) + 1.
        # mat = jnp.array([
        #     [s * jnp.cos(theta), -s * jnp.sin(theta)],
        #     [s * jnp.sin(theta), s * jnp.cos(theta)]
        # ]).transpose(2, 3, 0, 1)
        # k = jnp.squeeze(mat @ k0[:, None, :, None], -1)
        return k


class SNet(linen.Module):
    config: MaxwellModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        heads = int(features // 32)
        r_dim = r.shape[-1]

        r = jax.lax.stop_gradient(r)
        t = jax.lax.stop_gradient(t)

        eps_r = dielectric_fn(r)[..., 0:1]
        min_noise = jnp.finfo(self.config.dtype).resolution
        eps_min = eps_r.min()
        eps_scale = eps_r.max() - eps_r.min()
        x0 = jnp.where(eps_r.max() == eps_r.min(), jnp.zeros_like(eps_r), (eps_r - eps_min) / (eps_scale + min_noise))
        x0 = jnp.concatenate([x0, 1 - x0], axis=-1)

        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x)
        # x = linen.LayerNorm()(x)

        # delta_r = jnp.alltrue(jnp.equal(r, light_source.mu), axis=-1, keepdims=True)
        # delta_r = jnp.exp(-jnp.sum((r - light_source.mu) ** 2, axis=-1, keepdims=True) / 2 / (self.config.wavelength / 10) ** 2)
        # x0 = jnp.concatenate([x0, delta_r], axis=-1)

        r_emb = SpaceEmbedding(self.config)(r)
        t_emb = TimeEmbedding(self.config)(t)

        x = jnp.concatenate([x, r_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        x = jnp.concatenate([x, t_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x = NeuralGeneratorLayer(features)(x)

        x = linen.silu(linen.Dense(features)(x))
        s = linen.Dense(modes)(x)

        return s


class WNet(linen.Module):
    config: MaxwellModelConfig

    @linen.compact
    def __call__(self, h, r, t, light_source, dielectric_fn):
        features = self.config.features
        modes = self.config.modes
        heads = int(features // 32)
        r_dim = r.shape[-1]

        r = jax.lax.stop_gradient(r)
        t = jax.lax.stop_gradient(t)

        eps_r = dielectric_fn(r)[..., 0:1]
        min_noise = jnp.finfo(self.config.dtype).resolution
        eps_min = eps_r.min()
        eps_scale = eps_r.max() - eps_r.min()
        x0 = jnp.where(eps_r.max() == eps_r.min(), jnp.zeros_like(eps_r), (eps_r - eps_min) / (eps_scale + min_noise))
        x0 = jnp.concatenate([x0, 1 - x0], axis=-1)

        # envelope = jnp.exp(-(r[:, 0:1] - t) ** 2 / 2 / light_source.sigma_t ** 2)
        # x0 = jnp.concatenate([x0, envelope], axis=-1)

        # delta_r = jnp.alltrue(jnp.equal(r, light_source.loc), axis=-1, keepdims=True)
        # # delta_r = jnp.exp(-jnp.sum((r - light_source.mu) ** 2, axis=-1, keepdims=True) / 2 / (self.config.wavelength / 10) ** 2)
        # x0 = jnp.concatenate([x0, delta_r, 1 - delta_r], axis=-1)

        x = linen.silu(linen.Dense(features)(x0))
        x = linen.Dense(features)(x)

        r_emb = SpaceEmbedding(self.config)(r)
        t_emb = TimeEmbedding(self.config)(t)

        x = jnp.concatenate([x, r_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        x = jnp.concatenate([x, t_emb], -1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for _ in range(self.config.n_layers):
            x = NeuralGeneratorLayer(features)(x)

        x = linen.silu(linen.Dense(features)(x))
        w = linen.Dense(r_dim * modes)(x)

        # x1 = linen.Dense(features * 2)(x)
        # mu, logvar = linen.Dense(r_dim * modes)(x1[..., :features]), linen.Dense(r_dim * modes)(x1[..., features:])
        # var = jnp.exp(logvar)
        # w = jnp.exp(mu ** 2 / 2 / var)

        w = w.reshape(-1, r_dim, modes)

        # theta = SNet(self.config)(h, r, t, light_source, dielectric_fn)
        # w = jnp.stack([jnp.sin(theta), jnp.cos(theta)], 1) * w

        return w


class MaxwellModel(linen.Module):
    config: MaxwellModelConfig

    # flow: FlowTransform
    # prior: distrax.Distribution

    def init_state(self, rng):
        # W_h = jax.random.normal(rng, (self.config.mem_len, self.config.features // 2))
        # h = 2 * jnp.pi * W_h
        # h_i = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)

        W_h = jax.random.normal(rng, (self.config.mem_len, self.config.features))
        # g_h = jnp.linspace(-1., 1., self.config.mem_len)[:, None]
        # h_i = jnp.exp(g_h) * W_h
        h_i = jnp.sqrt(1 / self.config.r_dim) * W_h
        # h_i = jnp.sqrt(1 / self.config.features) * W_h
        # W_h = jax.random.uniform(rng, (self.config.mem_len, self.config.features))
        # h_i = 2 * (W_h - 0.5)

        # h_i = jnp.linspace(-1., 1., self.config.mem_len)[:, None]
        return h_i

    def setup(self):
        self.e_net = ENet(self.config)

    def get_observables(self, rng, h, r, t, light_source, dielectric_fn):
        c, hbar, m, q = self.config.c, self.config.hbar, self.config.m, self.config.q
        eps_0, wavelength = self.config.eps_0, self.config.wavelength
        omega = 2 * jnp.pi * c / wavelength

        source_r, source_t = light_source.sample(rng)
        # r_ls = light_source.mu
        # t_ls = t_l[:1]
        r = jnp.concatenate([r, source_r], 0)
        t = jnp.concatenate([t, source_t], 0)

        # r = r / au_const.nm * 1e-3
        # dL = 1 / au_const.nm * 1e-3
        # wavelength = wavelength / dL
        # omega =

        E_field = self.e_net(h, r, t, light_source, dielectric_fn)
        V = q * jnp.sum(r * E_field, axis=-1, keepdims=True)

        # def curl_E_fn(_r, _t):
        #     jac = jax.jacfwd(lambda some_r: self.e_net(h, some_r, _t, dielectric_fn))(_r)
        #     curl = jnp.stack([
        #         jac[..., 2, 1] - jac[..., 1, 2],
        #         jac[..., 0, 2] - jac[..., 2, 0],
        #         jac[..., 1, 0] - jac[..., 0, 1]
        #     ], axis=-1)
        #     return curl
        #
        # def curl_curl_E_fn(_r, _t):
        #     jac = jax.jacfwd(lambda some_r: curl_E_fn(some_r, _t))(_r)
        #     curl = jnp.stack([
        #         jac[..., 2, 1] - jac[..., 1, 2],
        #         jac[..., 0, 2] - jac[..., 2, 0],
        #         jac[..., 1, 0] - jac[..., 0, 1]
        #     ], axis=-1)
        #     return curl

        def grad_E_fn(_r, _t):
            _, grad_E = jax.jvp(lambda some_r: self.e_net(h, some_r, _t, light_source, dielectric_fn), [_r + 0j],
                                [jnp.ones_like(_r) + 0j])
            return grad_E

        def div_E_fn(_r, _t):
            _, grad_E = jax.jvp(lambda some_r: self.e_net(h, some_r, _t, light_source, dielectric_fn), [_r + 0j],
                                [jnp.ones_like(_r) + 0j])
            return grad_E.sum(axis=-1, keepdims=True)

        def lap_E_fn(_r, _t):
            _, lap_E = jax.jvp(lambda some_r: grad_E_fn(some_r, _t), [_r + 0j], [jnp.ones_like(_r) + 0j])
            return lap_E

        def dot_E_fn(_r, _t):
            _, dot_E = jax.jvp(lambda some_t: self.e_net(h, _r, some_t, light_source, dielectric_fn), [_t + 0j],
                               [jnp.ones_like(_t) + 0j])
            return dot_E

        def ddot_E_fn(_r, _t):
            _, ddot_E = jax.jvp(lambda some_t: dot_E_fn(_r, some_t), [_t + 0j], [jnp.ones_like(_t) + 0j])
            return ddot_E

        eps_r = dielectric_fn(r)
        ddot_E = ddot_E_fn(r, t)
        div_E = div_E_fn(r, t)
        # lap_E = jax.vmap(lap_E_fn)(r[:, None], t[:, None]).reshape(r.shape)
        lap_E = lap_E_fn(r, t)
        Ax = -lap_E - (1 / c) ** 2 * eps_r * ddot_E

        # Ax = -lap_E - (omega / c) ** 2 * eps_r * E_field

        # curl_curl_E = jax.vmap(curl_curl_E_fn)(r[:, None], t[:, None]).reshape(r.shape)
        # Ax = curl_curl_E - z * E_field

        def dot_J_fn(_r, _t):
            _, dot_J = jax.jvp(lambda some_t: light_source(_r, some_t), [_t + 0j], [jnp.ones_like(_t) + 0j])
            return dot_J

        # J = light_source(r, t)
        # b = (1 / eps_0 / c ** 2) * -1j * omega * J
        b = (1 / eps_0 / c ** 2) * dot_J_fn(r, t)
        err_em = Ax - b
        err_em = jnp.sum(jnp.real(err_em.conj() * err_em))
        err_em += jnp.sum(jnp.real(div_E.conj() * div_E))

        # def wf(_h, _r, _t):
        #     _, _p, _H = self.net(_h, _r, _t, potential_fn)
        #     if self.config.log_domain:
        #         return 1j * jnp.sum(_p * _r, axis=-1, keepdims=True) - 1j * _H * _t
        #     else:
        #         return jnp.exp(1j * jnp.sum(_p * _r, axis=-1, keepdims=True)) * jnp.exp(-1j * _H * _t)
        #
        # def dot_wf(_h, _r, _t):
        #     return jax.jacfwd(lambda some_t: wf(_h, _r, some_t), holomorphic=True)(_t + 0j)
        #
        # def grad_wf(_h, _r, _t):
        #     return jax.jacfwd(lambda some_r: wf(_h, some_r, _t), holomorphic=True)(_r + 0j)
        #
        # def lap_wf(_h, _r, _t):
        #     if self.config.log_domain:
        #         grad_log_psi = grad_wf(_h, _r, _t)
        #         _, lap_psi = jax.jvp(lambda some_r: grad_wf(_h, some_r, _t), [_r + 0j], [jnp.ones_like(_r) + 0j])
        #         return jnp.sum(lap_psi + grad_log_psi ** 2, axis=-1)
        #     else:
        #         _, lap_psi = jax.jvp(lambda some_r: grad_wf(_h, some_r, _t), [_r + 0j], [jnp.ones_like(_r) + 0j])
        #         return jnp.sum(lap_psi, axis=-1)
        #
        # L, D = r.shape
        # psi, vel, H_pred = self.net(h, r, t, potential_fn)
        # psi, h_next, vel, r_next, t_next = self.update(h, r, t, potential_fn)
        # p = vel * m

        # if self.config.log_domain:
        #     logq = 2 * psi
        # else:
        #     logq = jnp.log(psi.conj() * psi)

        # dot_psi = jax.vmap(dot_wf)(h[:, None], r[:, None], t[:, None])
        # dot_psi = dot_psi.reshape(L, 1)
        # lap_psi = jax.vmap(lap_wf)(h[:, None], r[:, None], t[:, None])
        # lap_psi = lap_psi.reshape(L, 1)
        # V = potential_fn(r)

        # if self.config.log_domain:
        #     # E = 1j * hbar * dot_psi
        #     H = -0.5 * hbar ** 2 / m * lap_psi + V
        #     # L = E - H
        # else:
        #     # E = 1j * hbar * dot_psi / psi
        #     H = -0.5 * hbar ** 2 / m * lap_psi / psi + V
        #     # psi_next, H_next = self.wf(psi, r_next, t_next, potential_fn)
        #     # lap_psi_next = jax.vmap(lap_wf)(psi[:, None], r_next[:, None], t_next[:, None])
        #     # lap_psi_next = lap_psi_next.reshape(L, 1)
        #     # V_next = potential_fn(r_next)
        #     # K_next = p_next ** 2 / (2 * m)
        #     # H_next = K_next + V_next
        #     # L = psi_next * jnp.exp(0.5j * dt * H_next) - psi * jnp.exp(-0.5j * dt * H)
        #     # L = E - H

        obs = dict(
            E_field=E_field,
            V=V,
            err_em=err_em,
            logq=0.
        )
        return obs

    def __call__(self, h, r, t, light_source, dielectric_fn):
        return self.e_net(h, r, t, light_source, dielectric_fn)

    def update(self, h, r, t, v, light_source, dielectric_fn):
        dt = self.config.dt

        r_next = r + v * dt
        t_next = t + dt
        v_next = v

        return h, r_next, t_next, v_next


def create(config: MaxwellModelConfig):
    hbar, m = config.hbar, config.m

    # layers = [NPILayer(config) for i in range(config.T)]
    # npi = NPI(config, layers)
    features, r_dim = config.features, config.r_dim

    # def mlp():
    #     return MLP([features, features, r_dim], act_fun=linen.tanh,
    #                kernel_init=linen.initializers.xavier_normal(), bias_init=linen.initializers.normal())
    #
    # transforms = [
    #     RealNVP(mlp(), False),
    #     RealNVP(mlp(), True),
    #     RealNVP(mlp(), False),
    #     RealNVP(mlp(), True),
    #     RealNVP(mlp(), False),
    #     RealNVP(mlp(), True),
    #     RealNVP(mlp(), False),
    #     RealNVP(mlp(), True),
    # ]
    #
    # flow = NormalizingFlow(transforms)
    # prior = distrax.Normal(jnp.zeros((r_dim,)), jnp.ones((r_dim,)))
    # prior = distrax.Independent(prior, 1)
    #
    # npi = NPI(config, flow, prior)

    model = MaxwellModel(config)

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

        # if config.use_ode:
        #
        #     def ode_step(_hr, _t, _params, _potential_fn):
        #         _h, _r = _hr
        #         _h_next, _vel, _, _ = npi.apply({'params': _params}, _h, _r, _t, _potential_fn, method=npi.update)
        #         return _h_next, _vel.real
        #
        #     ts = jnp.linspace(0., config.t_f, sample_length)
        #     ode_rtol = 1e-6
        #     h_traj, r_traj = jax.vmap(lambda _h, _r: odeint(ode_step, (_h, _r), ts, params, potential_fn, rtol=ode_rtol))(h_i[:, None], r_i[:, None])
        #
        #     h_traj = h_traj.reshape(sample_length, L, features)
        #     r_traj = r_traj.reshape(sample_length, L, r_dim)
        #     t_traj = jnp.tile(ts[:, None, None], (1, L, 1))
        #
        # else:

        def update_step(carry, key):
            _h, _r, _t, _v = carry
            _h_next, _r_next, _t_next, _v_next = model.apply({'params': params}, _h, _r, _t, _v, light_source, dielectric_fn, method=model.update)
            return (_h_next, _r_next, _t_next, _v_next), (_h_next, _r_next, _t_next, _v_next)

        _, (h_traj, r_traj, t_traj, v_traj) = jax.lax.scan(update_step, (h_i, r_i, t_i, v_i), keys)

        return h_traj, r_traj, t_traj, v_traj

    def get_observables(params, rngs, h_batch, r_batch, t_batch, light_source, dielectric_fn):
        def apply_fn(key, h, r, t):
            return model.apply({'params': params}, key, h, r, t,
                               light_source, dielectric_fn, method=model.get_observables)

        obs = jax.vmap(apply_fn, (0, 0, 0, 0))(rngs, h_batch, r_batch, t_batch)
        return obs

    def get_pred(params, h_batch, r_batch, t_batch, light_source, dielectric_fn):
        def apply_fn(h, r, t):
            return model.apply({'params': params}, h, r, t, light_source, dielectric_fn)

        pred = jax.vmap(apply_fn, (0, 0, 0))(h_batch, r_batch, t_batch)
        return pred

    def loss_fn(params, rng, ic, light_source, dielectric_fn, lamb=1.0):
        r_i, t_i, v_i = ic

        rng, key = jax.random.split(rng)
        h_i = model.init_state(rng)
        h_traj, r_traj, t_traj, v_traj = sample_train(params, key, h_i, r_i, t_i, v_i, light_source, dielectric_fn, lamb)
        # h_traj, r_traj, t_traj = data

        keys = jax.random.split(rng, h_traj.shape[0])
        obs_traj = get_observables(params, keys, h_traj, r_traj, t_traj, light_source, dielectric_fn)

        err_em = obs_traj['err_em']

        # V_traj = obs_traj['V']
        # V_traj1 = jnp.concatenate([V_i[None], V_traj], 0)
        # H_traj = obs_traj['H']
        # p_traj = obs_traj['p']
        # logq_traj = obs_traj['logq']

        # E = 0.5 * jnp.sum(p_i ** 2, axis=-1, keepdims=True) / m
        # V_diffs = V_traj1[1:] - V_traj1[:-1]
        # V_sum = jnp.cumsum(V_diffs, 0)
        # E = E_i + V_sum
        # err_pde = H_traj - E
        # err_pde = H_traj - E
        err_pde = jnp.sum(err_em)

        # p_pred_traj = obs_traj['p_pred']
        # err_score = p_traj - p_pred_traj
        # err_pde += jnp.mean(jnp.real(err_score.conj() * err_score))

        # psi_traj_prev = jnp.concatenate([jnp.ones((1, *t_i.shape)) * jnp.exp(0j), psi_traj[:-1]])
        # err_td_fwd = psi_traj - jnp.exp(-1j * dt * H_traj) * psi_traj_prev
        # err_td_bwd = psi_traj * jnp.exp(1j * dt * H_traj) - psi_traj_prev
        # err_td = jnp.mean(jnp.real(err_td_fwd.conj() * err_td_fwd)) + jnp.mean(jnp.real(err_td_bwd.conj() * err_td_bwd))
        # err_pde += err_td

        # E_i = jnp.mean(0.5 * jnp.sum(p_i ** 2, axis=-1, keepdims=True) / m + potential_fn(r_i))
        # E_i_pred = jnp.mean(H_traj, 1)
        # p_i_pred = p_traj[0]
        err_ic = 0.
        # err_ic = jnp.mean(jnp.sum(jnp.abs(p_i_pred - p_i) ** 2, axis=-1, keepdims=True))
        # err_ic += jnp.mean(jnp.abs(E_i_pred - E_i) ** 2)
        # err_ic = jnp.sum(jnp.abs(p_i_pred - p_i) ** 2, axis=-1, keepdims=True)
        # err_ic = jnp.mean(err_ic)

        # if config.log_domain:
        #     kldiv = jnp.mean(jnp.real(logq_traj - 2 * psi_traj))
        # else:
        #     kldiv = jnp.mean(jnp.real(logq_traj - jnp.log(psi_traj.conj() * psi_traj)))

        loss = err_pde + config.ic_weight * err_ic
        stats = dict(loss=loss, err_pde=err_pde, err_ic=err_ic)
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
