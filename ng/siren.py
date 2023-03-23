import jax
import jax.numpy as jnp
import optax
from flax import linen

from ng.train_state import TrainState


class SIRENLayer(linen.Module):
    features: int
    omega0: float = 1.

    @linen.compact
    def __call__(self, inputs):
        features = self.features
        omega0 = self.omega0

        def kernel_init(rng, shape, _):
            return jax.random.uniform(rng, shape, minval=-1., maxval=1.) * jnp.sqrt(6 / shape[0] / omega0 ** 2)

        kernel = self.param('kernel',
                            kernel_init,
                            (inputs.shape[-1], features),
                            inputs.dtype)

        bias = self.param('bias',
                          linen.initializers.zeros,
                          (features,),
                          inputs.dtype)

        y = jax.lax.dot_general(omega0 * inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))

        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return jnp.sin(y)


class SIREN(linen.Module):
    features: int
    n_classes: int = None
    n_layers: int = 5
    omega0: float = 1.
    out_dim: int = 1

    @linen.compact
    def __call__(self, x, label=None):
        features = self.features
        omega0 = self.omega0

        def kernel_init(rng, shape, _):
            return jax.random.uniform(rng, shape, minval=-1., maxval=1.) * jnp.sqrt(6 / shape[0])

        if label is not None:
            label = linen.one_hot(label, self.n_classes).reshape(-1, self.n_classes)
            x = jnp.concatenate([x, label], axis=-1)

        # x = jnp.sin(linen.Dense(features, kernel_init=kernel_init)(x) * omega0)

        for i in range(self.n_layers):
            x = SIRENLayer(features, omega0)(x)

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

        y = linen.Dense(self.out_dim)(x)
        return y


def create_siren(rng, batch, features, n_classes, lr):
    t, x, _ = batch

    model_def = SIREN(features, n_classes)
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
