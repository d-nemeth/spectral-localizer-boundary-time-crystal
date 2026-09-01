from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():

    plt.style.use("science")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
        }
    )

    RUN = "run_006"

    run_dir = Path("main_figures_datasets") / "topological_domains" / RUN

    figures_dir = run_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        run_dir / "info.json",
        "r",
    ) as f:
        info = json.load(f)

    data = np.load(
        run_dir / "data.npz",
        allow_pickle=True,
    )

    gammas = data["gammas"]

    x_arrays = data["x_arrays"]

    idx_arrays = data["idx_arrays"]

    selected_evals = data["selected_evals"]

    mode_idx = int(data["mode_idx"])

    kappa = float(data["kappa"])

    line_colours = [
        "#0072B2",
        "#D55E00",
        "#009E73",
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    lane_height = 3.2

    for j, gamma in enumerate(gammas):
        x = x_arrays[j]

        idx = idx_arrays[j]

        lam0 = selected_evals[j]

        y = idx + j * lane_height

        ax.step(
            x,
            y,
            where="mid",
            lw=1.8,
            color=line_colours[j],
        )

    for j in range(
        1,
        len(gammas),
    ):
        ax.axhline(
            j * lane_height,
            lw=0.8,
            alpha=0.25,
        )

    yticks = []

    yticklabels = []

    for j in range(len(gammas)):
        for val in [0, 2]:
            yticks.append(j * lane_height + val)

            yticklabels.append(str(val))

    ax.set_yticks(yticks)

    ax.set_yticklabels(yticklabels)

    ax.set_xlim(
        0.0,
        5.0,
    )

    ax.set_ylim(
        -0.2,
        lane_height * (len(gammas) - 1) + 2.6,
    )

    ax.set_xlabel(
        r"$x_0$",
    )

    ax.set_ylabel(
        r"$\nu^L$",
    )

    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=line_colours[j],
            lw=2,
        )
        for j in range(len(gammas))
    ]

    labels = [
        rf"$\tilde\Gamma="
        rf"{gamma:.1f}$"
        for gamma in gammas
    ]

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(gammas),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    plt.tight_layout(
        pad=0.3,
    )

    out_file = figures_dir / f"{RUN}.pdf"

    plt.savefig(
        out_file,
        bbox_inches="tight",
        dpi=400,
    )

    print(f"Saved figure:\n{out_file}")

    plt.show()


if __name__ == "__main__":
    main()
