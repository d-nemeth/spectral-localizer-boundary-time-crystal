from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import (
    ScalarMappable,
)
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
)

AXIS_LIMITS = {
    0: {
        "xlim": (-0.4, 0.01),
        "ylim": (-4.0, 4.0),
    },
    1: {
        "xlim": (-6.0, 0.5),
        "ylim": (-3.0, 3.0),
    },
}


def main():

    try:
        plt.style.use(["science"])

    except Exception:
        pass

    mpl.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
        }
    )

    RUN = "run_009"

    run_dir = Path("main_figures_datasets") / "topological_islands" / RUN

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

    params = info["params"]

    data = np.load(
        run_dir / "data.npz",
        allow_pickle=True,
    )

    labels = data["labels"]

    gammas = data["gammas"]

    x0_vals = data["x0_vals"]

    re_arrays = data["re_arrays"]

    im_arrays = data["im_arrays"]

    idx_arrays = data["idx_arrays"]

    eigvals_arrays = data["eigvals_arrays"]

    unique_idx_vals = data["unique_idx_vals"]

    idx_vals = unique_idx_vals.tolist()

    val_to_slot = {v: i for i, v in enumerate(idx_vals)}

    m = len(idx_vals)

    palette = [
        (0.94, 0.94, 0.94, 1.0),
        (0.15, 0.42, 0.38, 1.0),
        (0.20, 0.20, 0.20, 1.0),
        (0.55, 0.27, 0.27, 1.0),
        (0.36, 0.28, 0.50, 1.0),
        (0.45, 0.45, 0.20, 1.0),
    ]

    if m > len(palette):
        tmp = plt.get_cmap(
            "cividis",
            m,
        )

        colors = [tmp(i) for i in range(m)]

    else:
        colors = palette[:m]

    if 0 in val_to_slot:
        colors[val_to_slot[0]] = (
            0.94,
            0.94,
            0.94,
            1.0,
        )

    cmap = ListedColormap(colors)

    bounds = np.arange(
        -0.5,
        m + 0.5,
        1.0,
    )

    norm = BoundaryNorm(
        bounds,
        cmap.N,
    )

    contour_levels = np.arange(
        0.5,
        m,
        1.0,
    )

    def remap_ID(
        IDX: np.ndarray,
    ) -> np.ndarray:

        out = np.empty_like(
            IDX,
            dtype=int,
        )

        for v, s in val_to_slot.items():
            out[IDX == v] = s

        return out

    fig = plt.figure(
        figsize=(6.4, 3.2),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1,
            1,
            0.075,
        ],
        wspace=0.06,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]

    cax = fig.add_subplot(gs[0, 2])

    for i, ax in enumerate(axes):
        re_g = np.asarray(
            re_arrays[i],
            dtype=float,
        )

        im_g = np.asarray(
            im_arrays[i],
            dtype=float,
        )

        IDX = np.asarray(
            idx_arrays[i],
            dtype=int,
        )

        eigvals = np.asarray(
            eigvals_arrays[i],
            dtype=complex,
        )

        gamma = gammas[i]

        x0 = x0_vals[i]

        IDX_plot = remap_ID(IDX)

        extent = [
            re_g.min(),
            re_g.max(),
            im_g.min(),
            im_g.max(),
        ]

        ax.imshow(
            IDX_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
            zorder=0,
        )

        RR, II = np.meshgrid(
            re_g,
            im_g,
        )

        ax.contour(
            RR,
            II,
            IDX_plot,
            levels=contour_levels,
            colors=[
                (
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                )
            ],
            linewidths=0.6,
            antialiased=True,
            zorder=2,
        )

        re_min = re_g.min()
        re_max = re_g.max()

        im_min = im_g.min()
        im_max = im_g.max()

        mask = (
            (eigvals.real >= re_min)
            & (eigvals.real <= re_max)
            & (eigvals.imag >= im_min)
            & (eigvals.imag <= im_max)
        )

        xlim = AXIS_LIMITS.get(i, {}).get("xlim")

        ylim = AXIS_LIMITS.get(i, {}).get("ylim")

        if xlim is None:
            ax.set_xlim(
                re_min,
                re_max,
            )

        else:
            ax.set_xlim(*xlim)

        if ylim is None:
            ax.set_ylim(
                im_min,
                im_max,
            )

        else:
            ax.set_ylim(*ylim)

        ax.scatter(
            eigvals.real[mask],
            eigvals.imag[mask],
            s=10,
            alpha=0.8,
            facecolors="none",
            edgecolors=(
                0.05,
                0.05,
                0.05,
                0.55,
            ),
            linewidths=0.85,
            rasterized=True,
            zorder=3,
        )

        ax.axvline(
            0.0,
            color=(0, 0, 0, 0.18),
            lw=0.7,
        )

        ax.axhline(
            0.0,
            color=(0, 0, 0, 0.12),
            lw=0.7,
        )

        ax.text(
            0.04,
            0.95,
            rf"{labels[i]})",
            transform=ax.transAxes,
            fontsize=14,
            va="top",
        )

        ax.text(
            0.9,
            0.95,
            rf"$\tilde\Gamma={gamma:.2f}$",
            transform=ax.transAxes,
            fontsize=13,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": (1.0, 1.0, 1.0, 0.8),
                "edgecolor": (0.0, 0.0, 0.0, 0.1),
                "linewidth": 0.5,
            },
        )

        ax.minorticks_on()

        ax.tick_params(labelsize=14)

        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[0].set_ylabel(
        r"Im($\lambda_0$)",
        fontsize=15,
    )

    for ax in axes:
        ax.set_xlabel(
            r"Re($\lambda_0$)",
            fontsize=15,
        )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        ticks=np.arange(m),
    )

    cbar.ax.set_yticklabels([str(v) for v in idx_vals])

    cbar.set_label(
        r"$\nu^L$",
        fontsize=15,
    )

    cbar.outline.set_visible(False)

    cbar.ax.tick_params(labelsize=14)

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
