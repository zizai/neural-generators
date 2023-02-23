import math
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import linen, struct


def attention(query, key, value,
              mask=None,
              attn_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    k_dim = query.shape[-1]
    seq_len = value.shape[-2]

    if attn_type == 'cosine':
        denom = jnp.linalg.norm(query, axis=(-2, -1), keepdims=True) * jnp.linalg.norm(key, axis=(-2, -1), keepdims=True)
        p_attn = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / denom / math.sqrt(k_dim)
    else:
        scores = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / math.sqrt(k_dim)

        if attn_type == 'softmax':
            if mask is not None:
                scores = jnp.where(mask == 0, -1e9, scores)
            p_attn = jax.nn.softmax(scores, axis=-1)
        elif attn_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = jnp.where(mask == 0, 0., scores)
            p_attn = scores / seq_len
        else:
            raise ValueError

    out = jnp.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None,
                     attn_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = value.shape[-2]
    if attn_type in ['linear', 'global']:
        query = jax.nn.softmax(query, axis=-1)
        key = jax.nn.softmax(key, axis=-2)
    scores = jnp.matmul(key.transpose(0, 1, 3, 2), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    # if dropout is not None:
    #     p_attn = F.dropout(p_attn)

    out = jnp.matmul(query, p_attn)
    return out, p_attn


def causal_linear_attn(query, key, value, mask, eps=1e-8):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, k_dim = query.shape
    dtype = query.dtype

    key /= seq_len

    mask = mask[:, None, :, None]
    key = jnp.where(~mask, 0., key)
    value = jnp.where(~mask, 0., value)

    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, k_dim) for x in (query, key, value)]

    b_k_sum = b_k.sum(axis=-2)
    b_k_cumsum = b_k_sum.cumsum(axis=-2).astype(dtype)

    p_attn = jnp.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(axis=-3).astype(dtype)

    # if dropout is not None:
    #     p_attn = F.dropout(p_attn)

    D_inv = 1. / jnp.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = jnp.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn


class SimpleAttention(linen.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''
    attn_type: str
    features: int
    heads: int
    dtype: Any = jnp.float_
    symmetric_init: bool = False
    norm_eps: float = 1e-6

    def W_init(self, rng, shape, dtype):
        # p = linen.initializers.xavier_uniform()(rng, shape, dtype)
        p = linen.initializers.xavier_normal()(rng, shape, dtype)
        # p = linen.initializers.orthogonal(0.01)(rng, shape, dtype)
        if self.symmetric_init:
            p = 0.5 * (p + p.T)
        return p

    @linen.compact
    def __call__(self, query, key, value, pos=None, mask=None, weight=None):
        assert self.features % self.heads == 0
        k_dim = int(self.features / self.heads)

        if mask is not None:
            mask = mask[:, None, ...]

        B = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key

        fc = partial(linen.Dense, use_bias=False, kernel_init=self.W_init, dtype=self.dtype)
        weights = [fc(self.features) for _ in range(3)]

        query, key, value = \
            [W(x).reshape(B, self.heads, -1, k_dim)
             for W, x in zip(weights, (query, key, value))]

        ln = partial(linen.LayerNorm, epsilon=self.norm_eps, dtype=self.dtype)

        if self.attn_type in ['linear', 'galerkin', 'global']:
            norm_K = [ln() for _ in range(self.heads)]
            norm_V = [ln() for _ in range(self.heads)]

            key = jnp.stack(
                [norm(x) for norm, x in
                 zip(norm_K, (key[:, i, ...] for i in range(self.heads)))], axis=1)
            value = jnp.stack(
                [norm(x) for norm, x in
                 zip(norm_V, (value[:, i, ...] for i in range(self.heads)))], axis=1)
        else:
            norm_K = [ln() for _ in range(self.heads)]
            norm_Q = [ln() for _ in range(self.heads)]

            key = jnp.stack(
                [norm(x) for norm, x in
                 zip(norm_K, (key[:, i, ...] for i in range(self.heads)))], axis=1)
            query = jnp.stack(
                [norm(x) for norm, x in
                 zip(norm_Q, (query[:, i, ...] for i in range(self.heads)))], axis=1)

        if pos is not None:
            pos = pos[:, None, ...]
            pos = jnp.tile(pos, [1, self.heads, 1, 1])
            query, key = [jnp.concatenate([pos, x], axis=-1) for x in (query, key)]

        if self.attn_type in ['linear', 'galerkin', 'global']:
            x, attn_weight = linear_attention(query, key, value,
                                              mask=mask,
                                              attn_type=self.attn_type)
        elif self.attn_type == 'causal':
            assert mask is not None
            x, attn_weight = causal_linear_attn(query, key, value, mask)
        else:
            x, attn_weight = attention(query, key, value,
                                       mask=mask,
                                       attn_type=self.attn_type)

        out_dim = self.heads * k_dim if pos is None else self.heads * (k_dim + pos.shape[-1])
        att_output = x.reshape(B, -1, out_dim)

        if pos is not None:
            att_output = linen.Dense(self.features, dtype=self.dtype)(att_output)

        return att_output, attn_weight
