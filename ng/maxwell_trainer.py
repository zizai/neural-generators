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

from ng import maxwell_model
from ng.base_trainer import BaseTrainer
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
    def __init__(self, config: ConfigDict, init_cond: Callable, model_config: maxwell_model.MaxwellModelConfig, debug=False):
        super(MaxwellTrainer, self).__init__(config)
        self.model_config = model_config

        rng = jax.random.PRNGKey(self.seed)

        def get_ic(_key, n_samples=None):

            if n_samples is None:
                n_samples = config.n_samples

            r_i, t_i, v_i = init_cond(n_samples, _key)
            return r_i, t_i, v_i

        init, eval_step, sample_step, train_step = maxwell_model.create(model_config)

        rng, key = jax.random.split(rng)
        ic = get_ic(key)
        sno_def, params = init(key, ic, config.light_source, config.dielectric_fn)
        # print(jax.tree_map(lambda p: p.shape, params))

        opt = optax.adam(self.config.lr)
        state = TrainState.create(apply_fn=sno_def.apply, params=params, stats={}, opt=opt)

        self.rng = rng
        self.model_def = sno_def
        self.state = state
        self.get_ic = get_ic

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

    def eval(self, r_i, t_i, v_i):
        self.rng, key = jax.random.split(self.rng)
        h_i = self.model_def.init_state(key)

        hs, rs, vs, ts = [], [], [], []
        preds = []
        n_steps = int(self.model_config.t_f / self.model_config.dt)
        skip = n_steps / self.model_config.sample_length
        tqdm_iter = tqdm(range(n_steps))

        for step in tqdm_iter:
            self.rng, key1, key2, key3 = jax.random.split(self.rng, 4)
            h_next, r_next, t_next, v_next = self.sample_step(self.state.params, key2, h_i, r_i, t_i, v_i, self.config.light_source, self.config.dielectric_fn)

            if step % skip == 0:
                pred = self.eval_step(self.state.params, key1, h_i[None], r_i[None], t_i[None], self.config.light_source, self.config.dielectric_fn)

                preds.append(pred)
                rs.append(r_i)
                ts.append(t_i)
                vs.append(v_i)

            if step == n_steps - 1:
                pred = self.eval_step(self.state.params, key1, h_next[None], r_next[None], t_next[None], self.config.light_source, self.config.dielectric_fn)

                preds.append(pred)
                rs.append(r_next)
                ts.append(t_next)
                vs.append(v_next)

            h_i, r_i, t_i, v_i = h_next, r_next, t_next, v_next

        return onp.asarray(preds), onp.asarray(rs), onp.asarray(ts), onp.asarray(vs)

    def train(self, n_steps):
        tqdm_iter = tqdm(range(1, n_steps + 1))
        lambs = onp.linspace(0, 1, n_steps)
        # T = self.model_config.T
        # P = self.model_config.substeps
        # rb_size = T * P * 10

        for step in tqdm_iter:
            self.train_epoch += 1
            self.rng, *keys = jax.random.split(self.rng, 9)

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

            ic = self.get_ic(keys.pop())
            self.state, stats = self.train_step(self.state, keys.pop(), ic, self.config.light_source, self.config.dielectric_fn, lambs[step-1])

            loss = stats['loss']

            if onp.isnan(loss):
                print(stats)
                raise ValueError

            if loss < self.config.etol:
                break

            tqdm_iter.set_postfix(stats)
