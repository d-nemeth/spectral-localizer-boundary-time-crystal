# %%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    spectral_localizer,
    localizer_gap_and_index,
)

try:
    import scienceplots
    plt.style.use("science")
except Exception:
    pass

def main():
    # model setup
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    gamma_list = [0.0, 1.0, 2.0]
    lam0 = 0.0 + 0.0j

    Nx = 25 * (params.N_spins + 1)
    kappa = 1.0
    zero_tol = 1e-10

    x0_vals = np.linspace(0.0, 5.0, Nx)

    line_colours = ["#0072B2", "#D55E00", "#009E73"]

    # compute indices
    idx_data = {}
    for gamma in gamma_list:
        L_mat = build_L(gamma)
        idx_k = np.zeros(Nx, dtype=int)
        for i, x0 in enumerate(x0_vals):
            L_loc = spectral_localizer(L_mat, K_rank_mat, lam0=lam0, x0=float(x0), kappa=kappa)
            _, idx = localizer_gap_and_index(L_loc, zero_tol=zero_tol)
            idx_k[i] = idx
        idx_data[gamma] = idx_k

    # stacked plot
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lane_height = 3.2

    for j, gamma in enumerate(gamma_list):
        y = idx_data[gamma] + j * lane_height
        ax.step(x0_vals, y, where="mid", lw=1.8, color=line_colours[j])

    # separators between lanes
    for j in range(1, len(gamma_list)):
        ax.axhline(j * lane_height, lw=0.8, alpha=0.25)

    # repeated integer ticks
    yticks, yticklabels = [], []
    for j in range(len(gamma_list)):
        for val in [0, 2]:
            yticks.append(j * lane_height + val)
            yticklabels.append(str(val))

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-0.2, lane_height * (len(gamma_list) - 1) + 2.6)
    ax.set_xlabel(r"$x_0$", fontsize=15)
    ax.set_ylabel(r"$\nu^L$", fontsize=15)

    handles = [plt.Line2D([0], [0], color=line_colours[j], lw=2) for j in range(len(gamma_list))]
    labels = [rf"$\tilde\Gamma={g/(params.omega):.1f}$" for g in gamma_list]

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(gamma_list),
        frameon=False,
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout(pad=0.3)
    plt.show()

if __name__ == "__main__":
    main()
# %%