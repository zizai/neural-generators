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
from ng.materials import GenericMaterial
from ng.maxwell_model import MaxwellModelConfig
from ng.maxwell_trainer import maxwell_trainer_config, MaxwellTrainer
from ng.sources import DipoleSource, GaussianPulseSource


class DielectricPrism(GenericMaterial):
    rx: chex.Scalar
    ry: chex.Scalar
    x0: chex.Scalar = 0.
    y0: chex.Scalar = 0.
    alpha: chex.Scalar = 0.25 * np.pi
    eps_max: chex.Scalar = 1.512 ** 2

    def sample(self, rng, n_samples):
        r = jax.random.normal(rng, (n_samples, 2))
        r = jnp.stack([(r[..., 0] + self.x0) * self.rx, (r[..., 1] + self.y0) * self.ry], -1)
        return r

    def __call__(self, r, *args, **kwargs):
        eps_r = jnp.ones(r.shape)

        x, y = r[..., 0], r[..., 1]
        in_prism = jnp.logical_and(x >= self.x0, x <= self.rx - self.x0)
        in_prism = jnp.logical_and(in_prism, y >= self.y0)
        in_prism = jnp.logical_and(in_prism, x * jnp.tan(alpha) >= y)
        eps_r = jnp.where(in_prism[..., None], self.eps_max, eps_r)
        return eps_r


def run(init_sigma):
    seed = 47

    beta = 90 / 180 * jnp.pi
    wavelength = 1.5
    c = 1.
    omega = 2 * np.pi * c / wavelength
    eps_0 = 1.
    # t_f = t_f * wavelength
    Lx = 50
    Ly = 50

    t_i = 0.
    t_f = 3e2 * fs_l
    dt = 0.1 * fs_l

    source_E0 = jnp.array([[jnp.sin(beta), jnp.cos(beta)]]) * 1.
    source_k0 = 2 * jnp.pi / wavelength
    source_loc = (-wavelength, 0.)
    source_w = (None, None)
    source_t0 = 0.
    light_source = DipoleSource(source_loc, source_w, source_E0, source_k0, omega)
    # light_source = GaussianPulseSource(source_r, w_l / au_const.nm * 1e-3, source_t0, sigma_l / au_const.fs * fs_l,
    #                                    source_E0, source_k0, omega)

    trainer_config = maxwell_trainer_config()
    trainer_config.update(
        seed=seed,
        n_samples=args.n_samples,
        light_source=light_source,
        dielectric_fn=DielectricPrism(Lx, Ly, eps_max=eps_max),
        etol=1e-2
    )
    model_config = MaxwellModelConfig(
        t_i=t_i,
        t_f=t_f,
        dt=dt,
        sample_length=args.sample_length,
        c=c,
        eps_0=eps_0,
        wavelength=wavelength,
        omega=omega,
        E0_norm=E0_norm,
        init_sigma=init_sigma,
        features=args.features,
        modes=32,
        n_layers=args.n_layers,
        dtype=jnp.float32
    )

    def random_field_init(n_samples, rng):
        rng, *keys = jax.random.split(rng, 9)

        pos_x = jax.random.uniform(keys.pop(), (n_samples, 1), minval=-0.5 * Lx, maxval=1.5 * Lx)
        pos_y = jax.random.uniform(keys.pop(), (n_samples, 1), minval=-0.2 * Ly, maxval=0.8 * Ly)
        r_l = jnp.concatenate([pos_x, pos_y], axis=-1)
        assert r_l.shape == (n_samples, 2)

        # t_l = jnp.zeros((n_samples, 1)) + t_i
        t_l = jax.random.uniform(keys.pop(), (n_samples, 1)) * (t_f - t_i) + t_i
        v_l = jax.random.normal(keys.pop(), (n_samples, 2)) * 0.1 * c
        return r_l, t_l, v_l

    def grid_field_init(n_x, rng):
        n_y = n_x // 2
        pos_x = jnp.linspace(-0.5 * Lx, 1.5 * Lx, n_x)
        pos_y = jnp.linspace(-0.2 * Ly, 0.8 * Ly, n_y)
        r_l = jnp.stack(jnp.meshgrid(pos_x, pos_y), -1).reshape(-1, 2)
        t_l = jnp.zeros((n_x * n_y, 1))
        v_l = jax.random.normal(rng, (n_x * n_y, 2)) * 0.1 * c
        return r_l, t_l, v_l

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
    # ic = grid_field_init(20000, 20, trainer.rng)
    preds, rs, ts, vs = trainer.eval(*ic)
    E_field = preds[0]
    Ex = E_field[..., 0].reshape(100, 200)
    Ex_max = np.abs(Ex.real).max()
    divnorm = colors.TwoSlopeNorm(vmin=-Ex_max, vcenter=0., vmax=Ex_max)
    plt.imshow(np.flipud(Ex.real), cmap='RdBu', norm=divnorm)
    # plt.imshow(np.flipud(Ex.real), cmap='RdBu')
    plt.colorbar()
    # plt.show()
    plt.savefig(f'./prism_Ex_t_{ts[0].reshape(20000)[0]}_init_sigma_{init_sigma}_features_{args.features}.png')
    plt.clf()

    E_field = preds[-1]
    Ex = E_field[..., 0].reshape(100, 200)
    Ex_max = np.abs(Ex.real).max()
    divnorm = colors.TwoSlopeNorm(vmin=-Ex_max, vcenter=0., vmax=Ex_max)
    plt.imshow(np.flipud(Ex.real), cmap='RdBu', norm=divnorm)
    # plt.imshow(np.flipud(Ex.real), cmap='RdBu')
    plt.colorbar()
    # plt.show()
    plt.savefig(f'./prism_Ex_t_{ts[-1].reshape(20000)[0]}_init_sigma_{init_sigma}_features_{args.features}.png')
    plt.clf()

    # v_f = vs[-1]
    # beta_e_f = np.sqrt(np.sum(v_f ** 2, axis=-1)) / au_const.c
    # U_f = (np.sqrt(1 / (1 - beta_e_f ** 2)) - 1) * au_const.c ** 2 / au_const.eV
    # # U_f = np.sum(v_f ** 2, axis=-1) / 2
    # print(np.mean(np.sum(vs[0] ** 2, axis=-1) / 2))
    # data = U_f[:, None] - U0 / au_const.eV
    # data = pandas.DataFrame(data, columns=['dU'])
    # # sns.kdeplot(data=data, x='dU')
    # # plt.show()
    # sns.histplot(data=data, x='dU', bins=500)
    # plt.show()

    # r_f = rs[-1]
    # data = r_f / au_const.nm * 1e-3
    # data = pandas.DataFrame(data, columns=['x', 'y'])
    # sns.scatterplot(data=data, x='x', y='y', s=5)
    # plt.show()

    # r_traj = np.concatenate([rs[0], rs[-1]], 0)
    # t_traj = np.concatenate([ts[0], ts[-1]], 0)
    # data = np.concatenate([r_traj / au_const.nm * 1e-3, np.round(t_traj / fs_l, 2)], axis=1)
    # data = pandas.DataFrame(data, columns=['x', 'y', 't'])
    # sns.scatterplot(data=data, x='x', y='y', hue='t', s=5)
    # plt.show()

    # ic = grid_field_init(400, trainer.rng)
    # preds, rs, ts = trainer.eval(*ic)
    # eps_r = DielectricPrism(Lx, Ly)(rs[0])[:, 0]
    # plt.imshow(np.flipud(eps_r.reshape(nx, nx)), cmap='Greys')
    # plt.colorbar()
    # plt.show()
    # E_field = preds[0]['E_l']
    # Ex = E_field[..., 0].reshape(200, 400)
    # divnorm = colors.TwoSlopeNorm(vmin=-np.abs(Ex.real).max(), vcenter=0., vmax=np.abs(Ex.real).max())
    # plt.imshow(np.flipud(Ex.real), cmap='RdBu', norm=divnorm)
    # plt.colorbar()
    # plt.show()

    # Jx = J[..., 1].reshape(nx, nx)
    # plt.imshow(Jx.real, cmap='RdBu')
    # plt.colorbar()
    # plt.show()

    # r_i, r_f = r_traj[0], r_traj[-1]
    # x_i = r_i[:, 0].mean()
    # x_f = r_f[:, 0].mean()
    # v_pred = (x_f.mean() - center_x) / t_f
    # print(f'electron velocity (a.u.): {v_pred} (estimated) vs {beta * au_const.c} (truth)')
    # print(f'electron velecity (nm/fs): {v_pred / (au_const.nm / au_const.fs)} nm/fs')

    # skip = T * substeps // 10
    # plot_traj(rs[skip-1::skip, ::100] / au_const.nm, ts[skip-1::skip, ::100] / au_const.fs)

    # H_traj = obs_traj['H']
    # p_traj = obs_traj['p']
    # plot_p(p_traj[0].real, p_traj[-1].real)
    # plot_densities(p_traj[skip-1::skip].real)
    # plot_E(np.real(H_traj)[skip-1::skip].real, t_traj[skip-1::skip] / au_const.fs)


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=400)
    parser.add_argument('--sample_length', type=int, default=20)
    parser.add_argument('--train_steps', type=int, default=1000)
    args = parser.parse_args()

    fs_l = 1 / (1 / 3e8 * 1e-6 / 1e-15)

    U0 = 100e3 * au_const.eV
    E0_norm = 1e9 * au_const.V_m
    wavelength = 800 * au_const.nm
    omega = 2 * np.pi * au_const.c / wavelength
    k0 = 2 * np.pi / wavelength
    eps_max = 1.512 ** 2
    # eps_max = 1.
    l_range = np.arange(-100, 100)
    w_e = 100 * au_const.nm
    w_l = 30e3 * au_const.nm
    sigma_e = 20 * au_const.fs
    sigma_l = 20 * au_const.fs
    sigma_U = 200. * au_const.hbar * omega / au_const.eV
    eps_r = 1.512 ** 2
    alpha = 45 / 180 * np.pi

    beta_e = np.sqrt(1 - 1 / (U0 / au_const.m_e / au_const.c ** 2 + 1) ** 2)
    p = beta_e * au_const.c * au_const.m_e
    v_e = p / au_const.m_e
    v_l = au_const.c / np.sqrt(eps_r)

    cos_theta_match = 1 / np.sqrt(eps_r) / beta_e
    theta_match = np.arccos(cos_theta_match)
    phi_match = alpha - theta_match
    alpha2_match = np.arcsin(np.sqrt(eps_r) * np.sin(phi_match))

    alpha2 = alpha - args.beta / 180 * np.pi if args.beta else alpha2_match
    phi = np.arcsin(np.sin(alpha2) / np.sqrt(eps_r)) if args.beta else phi_match
    theta = alpha - phi if args.beta else theta_match

    sweep_sigmas = [0.1]
    for _sigma in sweep_sigmas:
        run(_sigma)
