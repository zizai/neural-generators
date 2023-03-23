import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as onp
import optax
import pandas
from flax import linen
from tqdm import tqdm

from ng.dnb import DNBLayer
from ng.siren import create_siren
from ng.train_state import TrainState


def sample_data(rng, n_samples, wl=0.5, scale=10):
    r = jax.random.uniform(rng, (n_samples, 1)) * scale
    return r, jnp.sin(2 * jnp.pi / wl / scale * r) / jnp.abs(r)


def sample_wavelengths(rng, n_wv):
    return jax.random.uniform(rng, (n_wv,))


class Dataset(object):
    def __init__(self, rng, size, n_classes):
        rng, key = jax.random.split(rng)

        self.wavelengths = sample_wavelengths(key, n_classes)
        n_samples = size // n_classes

        rng, *keys = jax.random.split(rng, n_classes + 1)
        labels = []
        ys = []
        ts = []

        for i, key in enumerate(keys):
            label = onp.zeros((n_samples, 1))
            label[:, :] = i
            t, y = sample_data(key, n_samples, self.wavelengths[i])

            labels.append(label)
            ys.append(y)
            ts.append(t)

        self.size = size
        self.ts = onp.concatenate(ts, 0)
        self.labels = onp.concatenate(labels, 0)
        self.ys = onp.concatenate(ys, 0)

    def sample(self, n_samples=500):
        choices = onp.random.choice([i for i in range(self.size)], size=n_samples, replace=False)
        return self.ts[choices], self.labels[choices], self.ys[choices]


class DNBNet(linen.Module):
    features: int
    n_classes: int
    mem_len: int = 100
    n_layers: int = 5

    def init_state(self, rng):
        h = jax.random.normal(rng, (self.mem_len, self.features)) * 0.1
        return h

    @linen.compact
    def __call__(self, h, t, label):
        features = self.features

        x = linen.one_hot(label, self.n_classes).reshape(-1, self.n_classes)

        # W_t = 2 ** jnp.linspace(-8, 0, features)
        # W_t = 2 * jnp.pi / W_t.reshape(1, features)
        # t_emb = jnp.concatenate([jnp.sin(t @ W_t), jnp.cos(t @ W_t)], -1)
        #
        # x = jnp.concatenate([x, t_emb], axis=-1)
        # x = linen.relu(linen.Dense(features * 2)(x))
        # x = linen.Dense(features)
        x = linen.relu(linen.Dense(features)(x))
        x = linen.Dense(features)(x) + x

        for i in range(self.n_layers):
            h, x = DNBLayer(features)(h, x)
            # x0 = x
            # x = linen.silu(linen.Dense(features)(x0))
            # x = linen.Dense(features)(x) + x0
            # # x = linen.LayerNorm()(x)

        x = linen.relu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        omega = linen.Dense(1)(x) * 2 * jnp.pi

        y = jnp.sin(omega * t)
        return y


class FourierMLP(linen.Module):
    features: int
    n_classes: int
    n_layers: int = 5

    @linen.compact
    def __call__(self, t, label):
        features = self.features

        x = linen.one_hot(label, self.n_classes).reshape(-1, self.n_classes)

        W_t = 2 ** jnp.linspace(-8, 8, features)
        W_t = 2 * jnp.pi / W_t.reshape(1, features)
        t_emb = jnp.concatenate([jnp.sin(t @ W_t), jnp.cos(t @ W_t)], -1)

        x = jnp.concatenate([x, t_emb], axis=-1)
        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)

        for i in range(self.n_layers):
            x0 = x
            x = linen.silu(linen.Dense(features)(x0))
            x = linen.Dense(features)(x) + x0

        x = linen.silu(linen.Dense(features * 2)(x))
        x = linen.Dense(features)(x)
        y = linen.Dense(1)(x)
        return y


def build_dnb_net(rng, batch, features, n_classes, lr):
    t, x, _ = batch

    model_def = DNBNet(features, n_classes)
    rng, key = jax.random.split(rng)
    h = model_def.init_state(key)
    rng, key = jax.random.split(rng)
    variables = model_def.init(key, h, t, x)
    params = variables['params']

    opt = optax.adam(lr)
    train_state = TrainState.create(apply_fn=model_def.apply, params=params, stats={}, opt=opt)

    def train_step(batch, state):
        t_batch, label_batch, y_batch = batch

        def loss_fn(_p, _t, _label, _y):
            _y_pred = model_def.apply({'params': _p}, h, _t, _label)
            loss = jnp.sum((_y_pred - _y) ** 2)
            return loss, dict(loss=loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, stats), grads = grad_fn(state.params, t_batch, label_batch, y_batch)
        state, updates = state.apply_gradients(grads=grads)
        return state, stats

    def eval_step(batch, state):
        t_batch, x_batch, y_batch = batch
        return model_def.apply({'params': state.params}, h, t_batch, x_batch)

    return train_state, train_step, eval_step


def build_fourier_mlp(rng, batch, features, n_classes, lr):
    t, x, _ = batch

    model_def = FourierMLP(features, n_classes)
    rng, key = jax.random.split(rng)
    variables = model_def.init(key, t, x)
    params = variables['params']

    opt = optax.adam(lr)
    train_state = TrainState.create(apply_fn=model_def.apply, params=params, stats={}, opt=opt)

    def train_step(batch, state):
        t_batch, label_batch, y_batch = batch

        def loss_fn(_p, _t, _label, _y):
            _y_pred = model_def.apply({'params': _p}, _t, _label)
            loss = jnp.sum((_y_pred - _y) ** 2)
            return loss, dict(loss=loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, stats), grads = grad_fn(state.params, t_batch, label_batch, y_batch)
        state, updates = state.apply_gradients(grads=grads)
        return state, stats

    def eval_step(batch, state):
        t_batch, x_batch, y_batch = batch
        return model_def.apply({'params': state.params}, t_batch, x_batch)

    return train_state, train_step, eval_step


def run():
    seed = 47
    batch_size = 1000
    data_size = 50000
    features = 256
    n_classes = 10
    lr = 1e-4
    train_steps = 10000

    rng = jax.random.PRNGKey(seed)

    rng, key = jax.random.split(rng)
    dataset = Dataset(key, data_size, n_classes)
    batch = dataset.sample(batch_size)

    rng, key = jax.random.split(rng)
    # train_state, train_step, eval_step = build_dnb_net(key, batch, features, n_classes, lr)
    # train_state, train_step, eval_step = build_fourier_mlp(key, batch, features, n_classes, lr)
    train_state, train_step, eval_step = create_siren(key, batch, features, n_classes, lr)
    train_step = jax.jit(train_step)
    eval_step = jax.jit(eval_step)

    eval_batch = dataset.sample(batch_size * n_classes)
    y_pred = eval_step(eval_batch, train_state)
    target_data = onp.concatenate([*eval_batch, jnp.ones(y_pred.shape)], -1)
    pred_data = onp.concatenate([*eval_batch[:2], y_pred, jnp.zeros(y_pred.shape)], -1)
    df = pandas.DataFrame(onp.concatenate([target_data, pred_data]), columns=['t', 'label', 'y', 'target'])
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=df, x='t', y='y', hue='label', style='target')
    plt.show()

    tqdm_iter = tqdm(range(1, train_steps + 1))
    omega_hist = []

    for step in tqdm_iter:
        batch = dataset.sample(batch_size)
        train_state, stats = train_step(batch, train_state)

        tqdm_iter.set_postfix(stats)

        # if step % 1000 == 0:
        #     batch = dataset.sample(batch_size * 2)
        #     _, omega_pred = eval_step(batch, train_state)
        #     result = onp.concatenate([onp.asarray(omega_pred), onp.ones(omega_pred.shape, dtype=onp.int32) * step], -1)
        #     omega_hist.append(result)

    eval_batch = dataset.sample(batch_size * n_classes)
    y_pred = eval_step(eval_batch, train_state)
    target_data = onp.concatenate([*eval_batch, jnp.ones(y_pred.shape)], -1)
    pred_data = onp.concatenate([*eval_batch[:2], y_pred, jnp.zeros(y_pred.shape)], -1)
    df = pandas.DataFrame(onp.concatenate([target_data, pred_data]), columns=['t', 'label', 'y', 'target'])
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=df, x='t', y='y', hue='label', style='target')
    plt.show()

    # omega_hist = onp.concatenate(omega_hist, 0)
    # df = pandas.DataFrame(omega_hist, columns=['omega', 'step'])
    # # sns.kdeplot(data=data, x='dU')
    # # plt.show()
    # sns.histplot(data=df, x='omega', hue='step')
    # plt.show()


if __name__ == '__main__':
    run()
