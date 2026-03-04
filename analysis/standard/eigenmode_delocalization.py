# %%
import numpy as np
import matplotlib.pyplot as plt

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    spectral_localizer,
    localizer_gap_and_index,
)

from spectral_localizer.kq_basis import build_kq_basis_from_casimir_and_Q
from spectral_localizer.mode_tools import (
    compute_rank_weights_norm_for_gamma,
    pick_three_modes_sorted,
)

def main():
    plt.style.use("science")

    index_color = "#1a1a1a"
    rank_color  = "#4E9A8E"

    # model setup
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    K2_mat, K_rank_mat, Q_mat = build_operator_space_coordinates(params)

    j = params.j

    x0_min, x0_max = 0.0, float(int(2 * j))
    Nx0 = 30 * (params.N_spins + 1)
    kappa = 1.0
    zero_tol = 1e-8

    # Build (k,q) basis
    kq_basis_vecs, kq_labels, k_values_present = build_kq_basis_from_casimir_and_Q(K2_mat, Q_mat, j)
    B = np.column_stack(kq_basis_vecs)  # orthonormal columns |k,q)

    # panel helper
    def plot_panel(ax1, L_mat, evals_sorted, mode_idx, k_list, rank_weights_norm, x0_vals, gamma):
        lam0_mode = complex(evals_sorted[mode_idx])

        prof_k = rank_weights_norm[mode_idx].copy()
        prof_k /= (prof_k.sum() + 1e-15)
        k_int = np.array(k_list, dtype=float)

        idx_vals = np.zeros_like(x0_vals, dtype=int)
        for i, x0 in enumerate(x0_vals):
            L_loc = spectral_localizer(L_mat, K_rank_mat, lam0=lam0_mode, x0=float(x0), kappa=kappa)
            _, idx = localizer_gap_and_index(L_loc, zero_tol=zero_tol)
            idx_vals[i] = idx

        l1 = ax1.step(x0_vals, idx_vals, where="mid", lw=2.0, color=index_color, zorder=3)
        ax1.plot(x0_vals, idx_vals, linestyle="none", marker="s", markersize=1.8, color=index_color, zorder=4)

        uniq = np.unique(idx_vals)
        ax1.set_yticks(uniq)
        ax1.set_yticklabels([str(int(v)) for v in uniq])

        ax1.tick_params(direction="in", labelsize=12)
        ax1.minorticks_off()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2 = ax1.twinx()
        l2 = ax2.plot(k_int, prof_k, "--o", lw=1.8, markersize=3.5, color=rank_color, zorder=5)
        ax2.tick_params(direction="in", labelsize=12)

        ax1.set_title(rf"$\lambda_0={lam0_mode.real:+.2f}{lam0_mode.imag:+.2f}i$", fontsize=15, pad=3)
        ax1.set_xlim(0.0, 10.0)
        ax1.set_ylim(0)
        ax2.set_ylim(0)
        return l1[0], l2[0], ax2

    # 2×3 figure
    gammas = [0.5, 1.5]
    x0_vals = np.linspace(x0_min, x0_max, Nx0)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True)

    for row, gamma in enumerate(gammas):
        L_mat = build_L(gamma)

        evals_sorted, rank_weights_norm_g, k_list_g = compute_rank_weights_norm_for_gamma(L_mat, B, kq_labels)
        modes = pick_three_modes_sorted(evals_sorted, Nslow=10, mid_frac=0.55, pick="large_real")

        for col, mode_idx in enumerate(modes):
            ax = axes[row, col]
            h1, h2, ax2 = plot_panel(ax, L_mat, evals_sorted, mode_idx, k_list_g, rank_weights_norm_g, x0_vals, gamma)

            if row == 1:
                ax.set_xlabel(r"$x_0$", fontsize=15)
            if col == 0:
                ax.set_ylabel(r"$\nu^L$", fontsize=15)

            # only label w_k on the last column
            if col == 2:
                ax2.set_ylabel(r"$w_k$", fontsize=15)

        axes[row, 0].text(
            -0.22, 0.5,
            rf"$\tilde\Gamma={gamma/params.omega:.1f}$",
            transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="center",
            fontsize=15
        )

    handles = [
        plt.Line2D([0], [0], lw=2.0, color=index_color, marker="s", markersize=4),
        plt.Line2D([0], [0], lw=1.4, color=rank_color, linestyle="--", marker="o", markersize=4),
    ]
    labels = [r"$\nu^L(x_0)$", r"$w_k$"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=15,
               handlelength=2.5, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

if __name__ == "__main__":
    main()
# %%