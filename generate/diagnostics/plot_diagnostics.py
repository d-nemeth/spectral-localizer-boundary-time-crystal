from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("science")
except Exception:
    pass


def main():

    # Load run data
    RUN = "run_004"

    run_dir = Path("supplemental_figures_datasets") / "localizer_scans" / RUN

    figures_dir = run_dir / "figures"

    figures_dir.mkdir(
        exist_ok=True,
    )

    # Load metadata
    with open(
        run_dir / "info.json",
        "r",
    ) as f:
        info = json.load(f)

    # Load data
    data = np.load(run_dir / "data.npz")

    x0_vals = data["x0_vals"]

    sweep_vals = data["sweep_vals"]

    mu_vals_all = data["mu_vals"]

    idx_vals_all = data["idx_vals"]

    # Metadata
    MODE = info["mode"]

    sweep_parameter = info["sweep_parameter"]

    # Styling
    mpl.rcParams.update(
        {
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
        }
    )

    # Figure layout
    nrows = len(sweep_vals)

    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(
            10,
            2.3 * nrows,
        ),
        sharex=True,
        gridspec_kw=dict(
            hspace=0.18,
            wspace=0.18,
        ),
    )

    # Colours
    index_color = "#0072B2"

    gap_color = "#D55E00"

    # Plot rows
    for i, sweep_val in enumerate(sweep_vals):
        ax_idx = axes[i, 0]

        ax_mu = axes[i, 1]

        mu_vals = mu_vals_all[i]

        idx_vals = idx_vals_all[i]

        # Labels
        if sweep_parameter == "gamma":
            sweep_label = (
                rf"$\Gamma="
                rf"{sweep_val:g}$"
            )

        elif sweep_parameter == "kappa":
            sweep_label = (
                rf"$\kappa="
                rf"{sweep_val:g}$"
            )

        else:
            sweep_label = f"{sweep_parameter}={sweep_val:g}"

        # Index
        ax_idx.step(
            x0_vals,
            idx_vals,
            where="mid",
            lw=2.0,
            color=index_color,
        )

        ax_idx.set_xlim(
            x0_vals.min(),
            3,
        )

        ax_idx.set_ylim(
            idx_vals.min() - 0.02,
            idx_vals.max() + 0.02,
        )

        ax_idx.set_ylabel(sweep_label + "\n" + r"$\nu^L$")

        # Gap
        ax_mu.plot(
            x0_vals,
            mu_vals,
            lw=2.0,
            color=gap_color,
        )

        ax_mu.set_xlim(
            x0_vals.min(),
            4.0,
        )

        ax_mu.set_ylim(
            mu_vals.min(),
            0.3,
        )

        ax_mu.set_ylabel(r"$\mu$")

    # Bottom labels
    axes[-1, 0].set_xlabel(r"$x_0$")

    axes[-1, 1].set_xlabel(r"$x_0$")

    # Clean axes
    for ax in axes.flatten():
        ax.grid(False)
        ax.set_axisbelow(True)

    # Save figure
    out_file = figures_dir / f"{RUN}.pdf"

    plt.savefig(
        out_file,
        bbox_inches="tight",
        dpi=400,
    )

    print(f"\nSaved figure:\n{out_file}")

    plt.show()


if __name__ == "__main__":
    main()
