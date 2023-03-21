import math

import jax
import jax.numpy as jnp
from flax import linen

from ng.attention import attention, linear_attention


class DynamicNeuralBasis(linen.Module):
    heads: int
    features: int
    mem_len: int = 256
    attn_type: str = 'local'
    layer_norm: bool = False

    @linen.compact
    def __call__(self, x, kv=None):
        assert self.features % self.heads == 0

        batch_size = 1 if x.ndim == 2 else x.shape[0]
        x_dim = x.shape[-1]
        k_dim = int(self.features / self.heads)

        if kv is not None:
            h_dim = kv.shape[-1]
        else:
            h_dim = self.mem_len

        Q = self.param('Q', linen.initializers.normal(1 / x_dim), (x_dim, self.features))
        K = self.param('K', linen.initializers.normal(1 / h_dim), (h_dim, self.features))
        V = self.param('V', linen.initializers.normal(1 / h_dim), (h_dim, self.features))
        g = self.param('log_scale', linen.initializers.zeros, (self.features,))

        query = x @ Q
        key = kv @ K if kv is not None else K
        value = jnp.exp(g) * (kv @ V) if kv is not None else jnp.exp(g) * V
        query, key, value = [d.reshape(batch_size, self.heads, -1, k_dim) for d in (query, key, value)]

        if self.layer_norm:
            query, key, value = [linen.LayerNorm()(d) for d in (query, key, value)]

        if self.attn_type == 'local':
            out, _ = attention(query, key, value, attn_type='fourier')
        elif self.attn_type == 'global':
            out, _ = linear_attention(query, key, value, attn_type='galerkin')
        elif self.attn_type == 'cosine':
            out, _ = attention(query, key, value, attn_type='cosine')
        else:
            out, _ = attention(query, key, value, attn_type='softmax')

        return jnp.reshape(out, (*x.shape[:-1], self.features))


class FourierNeuralBasis(linen.Module):
    attn_heads: int
    features: int
    mem_len: int = 256
    layer_norm: bool = False

    @linen.compact
    def __call__(self, x):
        batch_size = 1 if x.ndim == 2 else x.shape[0]
        x_dim = x.shape[-1]
        k_dim = self.features // self.attn_heads

        h_dim = self.features
        kv = self.param('kv', linen.initializers.normal(1.), (self.mem_len, h_dim))
        kv = jax.lax.stop_gradient(kv)
        modes = jnp.linspace(-2., 2., self.mem_len) * jnp.pi
        kv *= modes[:, None]

        log_scale = self.param('log_scale', linen.initializers.zeros, (self.mem_len, 1))
        scale = jnp.power(10., log_scale)

        Q = self.param('Q', linen.initializers.normal(1 / x_dim), (x_dim, self.features))
        K = self.param('K', linen.initializers.normal(1 / h_dim), (h_dim, self.features))
        V = self.param('V', linen.initializers.normal(1 / h_dim), (h_dim, self.features))

        query = x @ Q
        key = kv @ K
        value = kv @ V
        query, key, value = [d.reshape(batch_size, self.attn_heads, -1, k_dim) for d in (query, key, value)]

        if self.layer_norm:
            query, key = [linen.LayerNorm()(d) for d in (query, key)]

        c = jnp.matmul(query, key.transpose(0, 1, 3, 2))
        # c = jnp.matmul(jnp.cos(query), jnp.cos(key).transpose(0, 1, 3, 2))
        out = jnp.matmul(c, value) / self.mem_len
        out = out.reshape(-1, self.features)
        return out


class DNBLayer(linen.Module):
    features: int

    @linen.compact
    def __call__(self, h, x):
        features = self.features
        heads = int(features // 32)

        h = linen.silu(linen.Dense(features)(h))
        h = linen.Dense(features)(h) + h
        x = DynamicNeuralBasis(heads, features)(x, h) + x
        # x = linen.LayerNorm()(x)
        x0 = x
        x = linen.silu(linen.Dense(features)(x))
        x = linen.Dense(features)(x) + x0
        # x = linen.LayerNorm()(x)

        return h, x
