import math
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from ml_collections import ConfigDict
from scipy import constants
from tqdm import tqdm

from ng.base_trainer import BaseTrainer
from ng.maxwell_model import MaxwellModelConfig, create_maxwell_model
from ng.maxwell_potential_model import MaxwellPotentialModelConfig, create_maxwell_potential_model
from ng.train_state import TrainState


def maxwell_trainer_config():
    config = ConfigDict()

    config.name = 'Maxwell'
    config.seed = 0

    config.n_samples = 512
    config.light_source = None
    config.dielectric_fn = None

    config.optimizer = 'adam'
    config.lr = 1e-3
    config.etol = 1e-3
    return config


class MaxwellTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict, model_config: MaxwellPotentialModelConfig, debug=False):
        super(MaxwellTrainer, self).__init__(config)
        self.model_config = model_config

        rng = jax.random.PRNGKey(self.seed)
        self.rng = rng

        init, eval_step, sample_step, train_step = create_maxwell_potential_model(model_config)

        rng, key = jax.random.split(rng)
        ic = self.get_ic()
        sno_def, params = init(key, ic, config.light_source, config.dielectric_fn)
        # print(jax.tree_map(lambda p: p.shape, params))

        opt = optax.adam(self.config.lr)
        state = TrainState.create(apply_fn=sno_def.apply, params=params, stats={}, opt=opt)

        self.model_def = sno_def
        self.state = state

        if debug:
            self.eval_step = eval_step
            self.sample_step = sample_step
            self.train_step = train_step
        else:
            self.eval_step = jax.jit(eval_step)
            self.sample_step = jax.jit(sample_step)
            self.train_step = jax.jit(train_step)

        self.rb = []
        self.train_epoch = 0

    def get_ic(self, n_samples=None, scale=1.):
        self.rng, key = jax.random.split(self.rng)
        c = self.model_config.c
        x_domain, y_domain = self.model_config.x_domain, self.model_config.y_domain

        if n_samples is None:
            n_samples = self.config.n_samples

        n_x = round(math.sqrt((x_domain[1] - x_domain[0]) / (y_domain[1] - y_domain[0]) * n_samples))
        n_y = round(n_x * (y_domain[1] - y_domain[0]) / (x_domain[1] - x_domain[0]))
        pos_x = jnp.linspace(x_domain[0] * scale, x_domain[1] * scale, n_x)
        pos_y = jnp.linspace(y_domain[0] * scale, y_domain[1] * scale, n_y)
        r = jnp.stack(jnp.meshgrid(pos_x, pos_y), -1).reshape(-1, 2)
        pos_z = jnp.zeros((n_x * n_y, 1))
        r = jnp.concatenate([r, pos_z], -1)

        t = jnp.zeros((n_x * n_y, 1)) + self.model_config.t_domain[0]
        v = jax.random.normal(key, (n_x * n_y, 2)) * 0.1 * c
        v = jnp.concatenate([v, jnp.zeros((n_x * n_y, 1))], axis=-1)
        return r, t, v

    def eval(self, r_i, t_i, v_i):
        self.rng, key = jax.random.split(self.rng)
        h_i = self.model_def.init_state(key)

        hs, rs, vs, ts = [], [], [], []
        preds = []
        t_total = self.model_config.t_domain[1] - self.model_config.t_domain[0]
        n_steps = int(t_total / self.model_config.dt)
        skip = n_steps / self.model_config.sample_length
        tqdm_iter = tqdm(range(n_steps))

        for step in tqdm_iter:
            self.rng, key1, key2, key3 = jax.random.split(self.rng, 4)
            h_next, r_next, t_next, v_next = self.sample_step(self.state.params, key2, h_i, r_i, t_i, v_i, self.config.light_source, self.config.dielectric_fn)

            if step % skip == 0:
                pred = self.eval_step(self.state.params, key1, h_i, r_i, t_i, self.config.light_source, self.config.dielectric_fn)

                preds.append(pred)
                rs.append(r_i)
                ts.append(t_i)
                vs.append(v_i)

            if step == n_steps - 1:
                pred = self.eval_step(self.state.params, key1, h_next, r_next, t_next, self.config.light_source, self.config.dielectric_fn)

                preds.append(pred)
                rs.append(r_next)
                ts.append(t_next)
                vs.append(v_next)

            h_i, r_i, t_i, v_i = h_next, r_next, t_next, v_next

        return preds, onp.asarray(rs), onp.asarray(ts), onp.asarray(vs)

    def train(self, n_steps):
        tqdm_iter = tqdm(range(1, n_steps + 1))
        lambs = onp.linspace(0, 1, n_steps)
        # T = self.model_config.T
        # P = self.model_config.substeps
        # rb_size = T * P * 10

        for step in tqdm_iter:
            self.train_epoch += 1

            # if step % P == 0 or step == 1:
            #     psi_i, _, r_i, t_i = self.get_ic(keys.pop())
            #     samples = self.sample_step(self.state.params, keys.pop(), psi_i, r_i, t_i, self.config.dielectric_fn)
            #     r_traj, t_traj, mem_traj = jax.device_get(samples)
            #
            #     for sample_step in range(T * P):
            #         self.rb.append([r_traj[sample_step], t_traj[sample_step], mem_traj[sample_step]])
            #
            #     # self.rb.append((r_traj, t_traj, mem_traj))
            #
            #     if len(self.rb) > rb_size:
            #         self.rb = self.rb[-rb_size:]
            #
            # sample_i = onp.random.randint(0, len(self.rb), T)
            # samples = [self.rb[i] for i in sample_i]
            # r = onp.stack([d[0] for d in samples], 0)
            # t = onp.stack([d[1] for d in samples], 0)
            # mem = onp.stack([d[2] for d in samples], 0)
            # samples = r, t, mem

            self.rng, key = jax.random.split(self.rng)
            # ic = self.get_ic(scale=jnp.exp(2 * (step / n_steps - 1)))
            ic = self.get_ic(scale=step / n_steps + 0.1)

            self.state, stats = self.train_step(self.state, key, ic, self.config.light_source, self.config.dielectric_fn, lambs[step-1])

            loss = stats['loss']

            if onp.isnan(loss):
                print(stats)
                raise ValueError

            if loss < self.config.etol:
                break

            tqdm_iter.set_postfix(stats)
