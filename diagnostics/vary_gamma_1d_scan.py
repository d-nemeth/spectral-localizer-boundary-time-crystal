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
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.default": "regular",
    })

    # model setup
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    gamma_vals = np.linspace(0.0, 2.0, 5)

    lam0_fixed = 0.0 + 0.0j
    Nk = 20 * (params.N_spins + 1)
    kappa_k = 1.0
    zero_tol = 1e-8

    # build scan grid from eigenvalue ranges
    k_eigs = np.linalg.eigvalsh(K_rank_mat).real
    k0_vals = np.linspace(float(k_eigs.min()), float(k_eigs.max()), Nk)
    print(f"k0 range: [{k0_vals.min():.3f}, {k0_vals.max():.3f}]")

    for gamma in gamma_vals:
        L_mat = build_L(gamma)
        lam0 = lam0_fixed

        mu_k  = np.zeros(Nk, dtype=float)
        idx_k = np.zeros(Nk, dtype=int)

        for i, k0 in enumerate(k0_vals):
            L_loc = spectral_localizer(L_mat, K_rank_mat, lam0=lam0, x0=k0, kappa=kappa_k)
            mu, idx = localizer_gap_and_index(L_loc, zero_tol=zero_tol)
            mu_k[i]  = mu
            idx_k[i] = idx

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        axes[0].plot(k0_vals, mu_k, lw=1.2, marker='.', markersize=3)
        axes[0].set_ylim(1e-4, 1e0)
        axes[0].set_xlabel(r"$k_0$")
        axes[0].set_ylabel(r"$\mu(k_0)$")
        axes[0].set_title("Localizer gap vs $k_0$ (X = K_rank)")
        axes[0].grid(True)

        axes[1].step(k0_vals, idx_k, where="mid")
        axes[1].set_xlabel(r"$k_0$")
        axes[1].set_ylabel(r"$\nu^L$")
        axes[1].set_title("Localizer index vs $k_0$")
        axes[1].grid(True)

        plt.suptitle(
            f"Separate 1D spectral localizers | Gamma={gamma:.3f} | "
            f"lambda_0={lam0.real:.3f}+{lam0.imag:.3f}i",
            y=1.02
        )
        plt.tight_layout()
        plt.show()

    print("\nGamma scan done.")

if __name__ == "__main__":
    main()
# %%
