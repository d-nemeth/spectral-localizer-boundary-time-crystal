# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    spectral_localizer,
    localizer_gap_and_index,
)

def main():
    # style
    try:
        import scienceplots
        plt.style.use(["science"])
    except Exception:
        pass

    mpl.rcParams.update({
        "text.usetex": True,   # set False if LaTeX not installed
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

    gamma      = 1.0
    kappa_vals = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    X          = K_rank_mat
    x0         = 1.0
    zero_tol   = 1e-8

    # grids in complex plane
    re_grid = np.linspace(-0.15, 0.0, 20)
    im_grid = np.linspace(-1.5, 1.5, 20)

    # Build L and its spectrum (for overlay)
    L_mat = build_L(gamma)
    lam = np.linalg.eigvals(L_mat)

    # compute maps
    MU_all  = []
    IDX_all = []
    unique_vals_global = set()

    for kappa in kappa_vals:
        MU  = np.zeros((len(im_grid), len(re_grid)), dtype=float)
        IDX = np.zeros((len(im_grid), len(re_grid)), dtype=int)

        for yi, im0 in enumerate(im_grid):
            for xi, re0 in enumerate(re_grid):
                lam0 = complex(re0, im0)
                L_loc = spectral_localizer(L_mat, X, lam0=lam0, x0=x0, kappa=kappa)
                mu, idx = localizer_gap_and_index(L_loc, zero_tol=zero_tol)
                MU[yi, xi]  = mu
                IDX[yi, xi] = idx

        MU_all.append(MU)
        IDX_all.append(IDX)
        unique_vals_global |= set(np.unique(IDX).tolist())

        print(f"done κ={kappa:g} | ν unique={np.unique(IDX)}")

    MU_all  = np.array(MU_all)
    IDX_all = np.array(IDX_all)

    # colormaps
    palette = [
        (0.94, 0.94, 0.94, 1.0),  # 0
        (0.15, 0.42, 0.38, 1.0),  # +1
        (0.20, 0.20, 0.20, 1.0),  # -1
    ]
    val_to_color = {0: palette[0], 1: palette[1], -1: palette[2]}

    unique_vals_global = np.array(sorted(list(unique_vals_global)), dtype=int)
    colors_used = [val_to_color[v] for v in unique_vals_global]
    cmap_idx = ListedColormap(colors_used)

    bounds = np.concatenate([unique_vals_global - 0.5, [unique_vals_global[-1] + 0.5]])
    norm_idx = BoundaryNorm(bounds, cmap_idx.N)

    LOGMU_all = np.log10(np.maximum(MU_all, 1e-16))
    vmin_gap  = float(np.min(LOGMU_all))
    vmax_gap  = float(np.max(LOGMU_all))

    # plot
    nK = len(kappa_vals)
    fig, axes = plt.subplots(
        nK, 2,
        figsize=(9.5, 1.85*nK),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.2, wspace=0.08)
    )

    extent = [re_grid.min(), re_grid.max(), im_grid.min(), im_grid.max()]

    for i, kappa in enumerate(kappa_vals):
        axL = axes[i, 0]
        axR = axes[i, 1]

        imA = axL.imshow(
            IDX_all[i],
            origin="lower", aspect="auto",
            extent=extent,
            cmap=cmap_idx, norm=norm_idx,
            interpolation="nearest"
        )
        axL.scatter(
            lam.real, lam.imag,
            s=10, alpha=0.95,
            facecolors="none",
            edgecolors=(0.15, 0.15, 0.15, 0.8),
            linewidths=1.0
        )
        axL.set_ylabel(rf"$\kappa={kappa:g}$" + "\n" + r"Im($\lambda_0$)")
        axL.set_xlim(re_grid.min(), re_grid.max())
        axL.set_ylim(im_grid.min(), im_grid.max())

        imB = axR.imshow(
            LOGMU_all[i],
            origin="lower", aspect="auto",
            extent=extent,
            cmap="cividis",
            vmin=vmin_gap, vmax=vmax_gap,
            interpolation="bilinear"
        )
        axR.scatter(
            lam.real, lam.imag,
            s=10, alpha=0.95,
            facecolors="none",
            edgecolors=(0.15, 0.15, 0.15, 0.8),
            linewidths=1.0
        )
        axR.set_ylabel("")
        axR.set_xlim(re_grid.min(), re_grid.max())
        axR.set_ylim(im_grid.min(), im_grid.max())


    axes[-1, 0].set_xlabel(r"Re($\lambda_0$)")
    axes[-1, 1].set_xlabel(r"Re($\lambda_0$)")
    axes[0, 0].set_title(r"$\nu^L(\lambda_0)$", pad=8)
    axes[0, 1].set_title(r"$\log_{10}\mu(\lambda_0)$", pad=8)

    cbarA = fig.colorbar(
        imA, ax=axes[:, 0],
        orientation="horizontal",
        fraction=0.05, pad=0.08, location="top"
    )
    cbarA.set_ticks(unique_vals_global)
    cbarA.set_label(r"$\nu^L$")

    cbarB = fig.colorbar(
        imB, ax=axes[:, 1],
        orientation="horizontal",
        fraction=0.05, pad=0.08, location="top"
    )
    cbarB.set_label(r"$\log_{10}\mu$")
    cbarB.set_ticks([-5, -4, -3, -2, -1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%