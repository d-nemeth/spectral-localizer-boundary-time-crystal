# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    FastLocalizerConfig,
    fast_compute_idx_curve_for_gamma,
)

try:
    import scienceplots
    plt.style.use("science")
except Exception:
    pass


def main():
    # Model + coordinates
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    _, K_rank_mat, _ = build_operator_space_coordinates(params)


    # Scan settings
    gamma_list = [0.0, 1.0, 2.0]
    lam0 = 0.0 + 0.0j

    x_min, x_max = 0.0, 5.0

    cfg = FastLocalizerConfig(
        kappa=1.0,
        zero_tol=1e-10,
        n_coarse=100,
        max_refine=10,
        refine_only_changes=True,
        verbose=False,
    )

    # Parallel compute curves
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(fast_compute_idx_curve_for_gamma)(
            gamma,
            build_L=build_L,
            X=K_rank_mat,
            lam0=lam0,
            x_min=x_min,
            x_max=x_max,
            cfg=cfg,
        )
        for gamma in gamma_list
    )

    # Pack results keyed by gamma
    idx_data = {g: (x, idx) for (g, x, idx) in results}


    # Plot
    line_colours = ["#0072B2", "#D55E00", "#009E73"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lane_height = 3.2

    for j, gamma in enumerate(gamma_list):
        x, idx = idx_data[gamma]
        y = idx + j * lane_height
        ax.step(x, y, where="mid", lw=1.8, color=line_colours[j])

    # separators
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

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, lane_height * (len(gamma_list) - 1) + 2.6)
    ax.set_xlabel(r"$x_0$", fontsize=15)
    ax.set_ylabel(r"$\nu^L$", fontsize=15)

    handles = [
        plt.Line2D([0], [0], color=line_colours[j], lw=2)
        for j in range(len(gamma_list))
    ]
    labels = [
        rf"$\tilde\Gamma={g/params.omega:.1f}$"
        for g in gamma_list
    ]

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