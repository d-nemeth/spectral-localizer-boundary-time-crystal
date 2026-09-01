from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def main():

    try:
        plt.style.use("science")
    except Exception:
        pass

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

    run_name = "run_006"

    run_dir = Path("supplemental_figures_datasets") / "localizer_islands" / run_name

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(
        run_dir / "info.json",
        "r",
    ) as f:
        info = json.load(f)

    data = np.load(run_dir / "data.npz")

    re_grid = data["re_grid"]
    im_grid = data["im_grid"]
    kappa_vals = data["kappa_vals"]
    mu_vals = data["mu_vals"]
    idx_vals = data["idx_vals"]
    eigvals = data["eigvals"]

    unique_vals = np.array(
        sorted(
            set(
                map(
                    int,
                    np.unique(idx_vals),
                )
            )
        ),
        dtype=int,
    )

    palette = {
        0: (0.94, 0.94, 0.94, 1.0),
        1: (0.15, 0.42, 0.38, 1.0),
        -1: (0.20, 0.20, 0.20, 1.0),
        2: (0.55, 0.27, 0.27, 1.0),
        -2: (0.36, 0.28, 0.50, 1.0),
    }

    colors = [
        palette.get(
            int(v),
            plt.get_cmap("cividis")(i / max(len(unique_vals) - 1, 1)),
        )
        for i, v in enumerate(unique_vals)
    ]

    cmap_idx = ListedColormap(colors)

    bounds = np.concatenate(
        [
            unique_vals - 0.5,
            [unique_vals[-1] + 0.5],
        ]
    )

    norm_idx = BoundaryNorm(
        bounds,
        cmap_idx.N,
    )

    log_mu_vals = np.log10(
        np.maximum(
            mu_vals,
            1e-16,
        )
    )

    vmin_gap = float(np.min(log_mu_vals))

    vmax_gap = float(np.max(log_mu_vals))

    n_kappa = len(kappa_vals)

    fig, axes = plt.subplots(
        n_kappa,
        2,
        figsize=(9.5, 1.85 * n_kappa),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(
            hspace=0.2,
            wspace=0.18,
        ),
    )

    if n_kappa == 1:
        axes = np.array([axes])

    extent = [
        re_grid.min(),
        re_grid.max(),
        im_grid.min(),
        im_grid.max(),
    ]

    im_idx = None
    im_gap = None

    for i, kappa in enumerate(kappa_vals):
        ax_idx = axes[i, 0]
        ax_gap = axes[i, 1]

        im_idx = ax_idx.imshow(
            idx_vals[i],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_idx,
            norm=norm_idx,
            interpolation="nearest",
        )

        ax_idx.scatter(
            eigvals.real,
            eigvals.imag,
            s=10,
            alpha=0.95,
            facecolors="none",
            edgecolors=(0.15, 0.15, 0.15, 0.8),
            linewidths=1.0,
        )

        ax_idx.set_ylabel(rf"$\kappa={kappa:g}$" + "\n" + r"Im($\lambda_0$)")

        im_gap = ax_gap.imshow(
            log_mu_vals[i],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="cividis",
            vmin=vmin_gap,
            vmax=vmax_gap,
            interpolation="bilinear",
        )

        ax_gap.scatter(
            eigvals.real,
            eigvals.imag,
            s=10,
            alpha=0.95,
            facecolors="none",
            edgecolors=(0.15, 0.15, 0.15, 0.8),
            linewidths=1.0,
        )

        ax_gap.set_ylabel("")

        for ax in (ax_idx, ax_gap):
            ax.set_xlim(
                re_grid.min(),
                re_grid.max(),
            )

            ax.set_ylim(
                im_grid.min(),
                im_grid.max(),
            )

    axes[-1, 0].set_xlabel(r"Re($\lambda_0$)")

    axes[-1, 1].set_xlabel(r"Re($\lambda_0$)")

    axes[0, 0].set_title(
        r"$\nu^L(\lambda_0)$",
        pad=8,
    )

    axes[0, 1].set_title(
        r"$\log_{10}\mu(\lambda_0)$",
        pad=8,
    )

    cbar_idx = fig.colorbar(
        im_idx,
        ax=axes[:, 0],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        location="top",
    )

    cbar_idx.set_ticks(unique_vals)

    cbar_idx.set_label(r"$\nu^L$")

    cbar_gap = fig.colorbar(
        im_gap,
        ax=axes[:, 1],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        location="top",
    )

    cbar_gap.set_label(r"$\log_{10}\mu$")

    out_file = figures_dir / f"{run_name}.pdf"

    plt.savefig(
        out_file,
        bbox_inches="tight",
        dpi=400,
    )

    print(f"\nSaved figure:\n{out_file}")

    plt.show()


if __name__ == "__main__":
    main()
