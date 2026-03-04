#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    spectral_localizer,
    localizer_gap_and_index,
)

def main():
    try:
        import scienceplots
        plt.style.use("science")
    except Exception:
        pass

    # Font + tick styling
    mpl.rcParams.update({
        "text.usetex": True,
        "mathtext.default": "regular",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    # model setup
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    gamma = 1.0
    kappa_vals = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    lam0 = 0.0 + 0.0j
    zero_tol = 1e-8
    Nk = 20 * (params.N_spins + 1)

    # coordinate grid
    k_eigs = np.linalg.eigvalsh(K_rank_mat).real
    k0_vals = np.linspace(float(k_eigs.min()), float(k_eigs.max()), Nk)

    L_mat = build_L(gamma)

    mu_all, idx_all = [], []
    for kappa_k in kappa_vals:
        mu_k  = np.zeros(Nk)
        idx_k = np.zeros(Nk, dtype=int)

        for i, k0 in enumerate(k0_vals):
            L_loc = spectral_localizer(L_mat, K_rank_mat, lam0=lam0, x0=k0, kappa=kappa_k)
            mu, idx = localizer_gap_and_index(L_loc, zero_tol=zero_tol)
            mu_k[i], idx_k[i] = mu, idx

        mu_all.append(mu_k)
        idx_all.append(idx_k)

    mu_all  = np.array(mu_all)
    idx_all = np.array(idx_all)

    # Vertical layout
    nK = len(kappa_vals)
    fig, axes = plt.subplots(
        nK, 2,
        figsize=(10, 2.3*nK),
        sharex=True,
        gridspec_kw=dict(hspace=0.18, wspace=0.18)
    )

    # Better colours
    index_color = "#0072B2"   # blue
    gap_color   = "#D55E00"   # vermillion

    for i, kappa_k in enumerate(kappa_vals):
        ax_idx = axes[i, 0]
        ax_mu  = axes[i, 1]

        ax_idx.step(k0_vals, idx_all[i], where="mid", lw=2.0, color=index_color)
        ax_idx.set_xlim(0, 5)
        ax_idx.set_xticks([0, 2.5, 5])
        ax_idx.set_ylim(-0.02, 1.02)
        ax_idx.set_yticks([0, 1])
        ax_idx.set_ylabel(rf"$\kappa={kappa_k:g}$" + "\n" + r"$\nu^L$")

        ax_mu.plot(k0_vals, mu_all[i], lw=2.0, color=gap_color)
        ax_mu.set_xlim(0, 5)
        ax_mu.set_xticks([0, 2.5, 5])
        ax_mu.set_ylim(0, 0.3)
        ax_mu.set_yticks([0, 0.1, 0.2, 0.3])
        ax_mu.set_ylabel(r"$\mu$")

    axes[-1, 0].set_xlabel(r"$x_0$")
    axes[-1, 1].set_xlabel(r"$x_0$")

    for ax in axes.flatten():
        ax.grid(False)

    plt.tight_layout()
    # FILE_NAME = f"localizer_kappa_scan_N_{params.N_spins}_gamma_{gamma}"
    # plt.savefig(FILE_NAME + ".png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
# %%
