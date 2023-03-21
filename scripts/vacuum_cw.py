import math

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from matplotlib import colors
from scipy import integrate
from scipy.special import jv
from tqdm import trange, tqdm

from ng import au_const
from ng.materials import GenericMaterial, DielectricVacuum
from ng.maxwell_model import MaxwellModelConfig
from ng.maxwell_potential_model import MaxwellPotentialModelConfig
from ng.maxwell_trainer import maxwell_trainer_config, MaxwellTrainer
from ng.sources import DipoleSource


def plot_fields(E, file_prefix='vacuum_cw'):
    Ex = E[..., 0]
    Ex_max = np.abs(Ex.real).max()
    divnorm = colors.TwoSlopeNorm(vmin=-Ex_max, vcenter=0., vmax=Ex_max)
    plt.imshow(np.flipud(Ex.real), cmap='RdBu', norm=divnorm)
    # plt.imshow(np.flipud(Ex.real), cmap='RdBu')
    plt.colorbar()
    # plt.show()
    plt.savefig('./' + file_prefix + '_Ex.png')
    plt.clf()

    Ey = E[..., 1]
    Ey_max = np.abs(Ey.real).max()
    divnorm = colors.TwoSlopeNorm(vmin=-Ey_max, vcenter=0., vmax=Ey_max)
    plt.imshow(np.flipud(Ey.real), cmap='RdBu', norm=divnorm)
    # plt.imshow(np.flipud(Ex.real), cmap='RdBu')
    plt.colorbar()
    # plt.show()
    plt.savefig('./' + file_prefix + '_Ey.png')
    plt.clf()


def run(init_sigma):
    seed = 47

    beta = 90 / 180 * jnp.pi
    wavelength = 1.5
    c = 1.
    omega = 2 * np.pi * c / wavelength
    eps_0 = 1.
    k0 = 2 * np.pi / wavelength
    # t_f = t_f * wavelength
    Lx = 50
    Ly = 50

    t_domain = (0., wavelength / c * fs_l)
    x_domain = (-0.5 * Lx, 1.5 * Lx)
    y_domain = (-0.2 * Ly, 0.8 * Ly)
    dt = 0.1 * fs_l
    E0 = 1.

    source_E0 = jnp.array([[jnp.sin(beta), jnp.cos(beta), 0.]]) * E0
    source_k0 = jnp.array([[jnp.cos(beta), jnp.sin(beta), 0.]]) * k0
    source_loc = (0., 0., 0.)
    source_w = (None, None, None)
    source_t0 = 0.
    light_source = DipoleSource(source_loc, source_w, source_E0, k0, omega, t_domain[0], t_domain[1])
    # light_source = GaussianPulseSource(source_r, w_l / au_const.nm * 1e-3, source_t0, sigma_l / au_const.fs * fs_l,
    #                                    source_E0, source_k0, omega)

    trainer_config = maxwell_trainer_config()
    trainer_config.update(
        seed=seed,
        n_samples=args.n_samples,
        light_source=light_source,
        dielectric_fn=DielectricVacuum(),
        etol=1e-2
    )
    model_config = MaxwellPotentialModelConfig(
        t_domain=t_domain,
        x_domain=x_domain,
        y_domain=y_domain,
        dt=dt,
        sample_length=args.sample_length,
        c=c,
        eps_0=eps_0,
        init_sigma=init_sigma,
        features=args.features,
        modes=1,
        n_layers=args.n_layers,
        dtype=jnp.float32
    )

    def random_field_init(n_samples, rng):
        rng, *keys = jax.random.split(rng, 9)

        pos_x = jax.random.uniform(keys.pop(), (n_samples, 1), minval=x_domain[0], maxval=x_domain[1])
        pos_y = jax.random.uniform(keys.pop(), (n_samples, 1), minval=y_domain[0], maxval=y_domain[1])
        pos_z = jnp.zeros((n_samples, 1))
        r = jnp.concatenate([pos_x, pos_y, pos_z], axis=-1)

        # t = jnp.zeros((n_samples, 1)) + t_i
        t = jax.random.uniform(keys.pop(), (n_samples, 1)) * (t_domain[1] - t_domain[0]) + t_domain[0]
        v = jax.random.normal(keys.pop(), (n_samples, 2)) * 0.1 * c
        v = jnp.concatenate([v, jnp.zeros((n_samples, 1))], axis=-1)
        return r, t, v

    def grid_field_init(n_x, rng):
        n_y = n_x // 2
        pos_x = jnp.linspace(x_domain[0], x_domain[1], n_x)
        pos_y = jnp.linspace(y_domain[0], y_domain[1], n_y)
        r = jnp.stack(jnp.meshgrid(pos_x, pos_y), -1).reshape(-1, 2)
        pos_z = jnp.zeros((n_x * n_y, 1))
        r = jnp.concatenate([r, pos_z], -1)

        t = jnp.zeros((n_x * n_y, 1))
        v = jax.random.normal(rng, (n_x * n_y, 2)) * 0.1 * c
        v = jnp.concatenate([v, jnp.zeros((n_x * n_y, 1))], axis=-1)
        return r, t, v

    trainer = MaxwellTrainer(trainer_config, random_field_init, model_config, debug=False)
    # obs_traj, r_traj, t_traj = trainer.eval()
    # r_i, r_f = r_traj[0], r_traj[-1]
    # x_i = r_i[:, 0].mean()
    # x_f = r_f[:, 0].mean()
    # v_pred = (x_f.mean() - center_x) / t_f
    # print(f'electron velocity (a.u.): {v_pred} (estimated) vs {beta * au_const.c} (truth)')
    # print(f'electron velecity (nm/fs): {v_pred / (au_const.nm / au_const.fs)} nm/fs')

    trainer.train(args.train_steps)

    ic = grid_field_init(200, trainer.rng)
    ic_r, ic_t, _ = ic
    ic_E = light_source.get_fields(ic_r, ic_t)
    ic_E = ic_E.reshape(100, 200, -1)
    plot_fields(ic_E, 'vacuum_cw_ic')

    preds, rs, ts, vs = trainer.eval(*ic)
    E_pred = preds[0].reshape(100, 200, -1)
    plot_fields(E_pred, f'vacuum_cw_t_{ts[0].reshape(20000)[0]}_init_sigma_{init_sigma}_features_{args.features}')


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", False)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=400)
    parser.add_argument('--sample_length', type=int, default=10)
    parser.add_argument('--train_steps', type=int, default=1000)
    args = parser.parse_args()

    fs_l = 1 / (1 / 3e8 * 1e-6 / 1e-15)

    sweep_sigmas = [0.1]
    for s in sweep_sigmas:
        run(s)
