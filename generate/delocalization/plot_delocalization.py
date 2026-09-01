from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_panel(
    ax,
    k_list,
    rank_profile,
):

    prof_k = np.array(
        rank_profile,
        dtype=float,
    )

    prof_k /= prof_k.sum() + 1e-15

    ax.bar(
        k_list,
        prof_k,
        width=0.65,
        color="#222222",
        alpha=0.92,
    )

    ax.set_xlim(-0.5, 5.5)

    ax.set_ylim(0, 0.6)

    ax.set_xticks(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    )

    ax.set_yticks([0, 0.2, 0.4, 0.6])

    ax.tick_params(
        direction="out",
        length=3,
        width=0.8,
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.4,
        alpha=0.15,
    )

    ax.set_axisbelow(True)


def main():

    plt.style.use("science")

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.8,
        }
    )

    RUN = "run_006"

    run_dir = Path("main_figures_datasets") / "delocalization" / RUN

    fig_dir = run_dir / "figures"

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        run_dir / "info.json",
        "r",
    ) as f:
        info = json.load(f)

    gammas = info["gammas"]

    mode_idx = info["mode_idx"]

    data = np.load(
        run_dir / "data.npz",
        allow_pickle=True,
    )

    k_list = data["k_list"]

    evals_all = data["evals"]

    rank_weights_R_all = data["rank_weights_R"]

    rank_weights_L_all = data["rank_weights_L"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.4, 4.8),
        dpi=200,
        sharex=True,
        sharey=True,
    )

    panel_labels = [
        "a)",
        "b)",
        "c)",
        "d)",
    ]

    panel_counter = 0

    for row, gamma in enumerate(gammas):
        evals_sorted = evals_all[row]

        rank_weights_R = rank_weights_R_all[row]

        rank_weights_L = rank_weights_L_all[row]

        # Left profile
        plot_panel(
            axes[row, 0],
            k_list,
            rank_weights_L[mode_idx],
        )

        axes[row, 0].text(
            0.04,
            0.93,
            panel_labels[panel_counter],
            transform=axes[row, 0].transAxes,
            fontsize=13,
            va="top",
        )

        axes[row, 0].text(
            0.96,
            0.93,
            rf"$\tilde\Gamma={gamma:.1f}$",
            transform=axes[row, 0].transAxes,
            fontsize=13,
            ha="right",
            va="top",
        )

        panel_counter += 1

        # Right profile
        plot_panel(
            axes[row, 1],
            k_list,
            rank_weights_R[mode_idx],
        )

        axes[row, 1].text(
            0.04,
            0.93,
            panel_labels[panel_counter],
            transform=axes[row, 1].transAxes,
            fontsize=13,
            va="top",
        )

        axes[row, 1].text(
            0.96,
            0.93,
            rf"$\tilde\Gamma={gamma:.1f}$",
            transform=axes[row, 1].transAxes,
            fontsize=13,
            ha="right",
            va="top",
        )

        panel_counter += 1

    axes[1, 0].set_xlabel(r"$k$")

    axes[1, 1].set_xlabel(r"$k$")

    axes[0, 0].set_ylabel(r"$w_k^{(L)}$")

    axes[1, 0].set_ylabel(r"$w_k^{(L)}$")

    axes[0, 1].set_ylabel(r"$w_k^{(R)}$")

    axes[1, 1].set_ylabel(r"$w_k^{(R)}$")

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.98,
        wspace=0.22,
        hspace=0.22,
    )

    out_file = fig_dir / f"{RUN}.pdf"

    plt.savefig(out_file, bbox_inches="tight", dpi=400)

    print(f"Saved figure:\n{out_file}")

    plt.show()


if __name__ == "__main__":
    main()
