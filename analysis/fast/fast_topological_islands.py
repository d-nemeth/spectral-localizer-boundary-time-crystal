# %%
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import pickle
import contextlib
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
)
from spectral_localizer.fast_localizer import localizer_index_ldl


# Style
try:
    import scienceplots
    plt.style.use(["science"])
except Exception:
    pass

mpl.rcParams.update({
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
})



# Checkpoint helpers
CHECKPOINT_DIR = Path("checkpoints_localizer")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_PATH = CHECKPOINT_DIR / f"localizer_panels_{RUN_TAG}.pkl"


def save_checkpoint(path: Path, payload: dict):
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


@contextlib.contextmanager
def tqdm_joblib_with_panel_eta(tqdm_object, total_panels: int, t_start: float):
    """
    Make joblib update tqdm, plus ETA.
    """
    from joblib.parallel import BatchCompletionCallBack
    import joblib.parallel

    old_cb = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallBack(old_cb):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)

            done = tqdm_object.n
            elapsed = time.perf_counter() - t_start
            rate = done / max(elapsed, 1e-12)
            eta = (total_panels - done) / max(rate, 1e-12)
            tqdm_object.set_postfix_str(f"saved {done}/{total_panels} | ETA {eta/60:.1f} min")
            return super().__call__(*args, **kwargs)

    try:
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


# Fast panel precomputation
class PanelLocalizerPrecomp:
    """
    Localizer for fixed (L, X, x0, kappa) but varying lam0.
    Builds:
        M(lam0) = [[ kappa(X - x0 I),  L - lam0 I ],
                  [ (L - lam0 I)†  , -kappa(X - x0 I) ]]
    and updates lam0 by O(N) diagonal updates in TR/BL blocks.
    """

    def __init__(self, L_mat: np.ndarray, X: np.ndarray, x0: float, kappa: float):
        self.N = int(L_mat.shape[0])
        self.kappa = float(kappa)
        self.x0 = float(x0)

        N = self.N
        Xh = 0.5 * (X + X.conj().T)

        TL = self.kappa * (Xh - self.x0 * np.eye(N, dtype=complex))
        BR = -TL

        TR0 = L_mat.astype(complex, copy=False)  # this is L (we'll subtract lam0 on diag)
        BL0 = TR0.conj().T

        M = np.empty((2 * N, 2 * N), dtype=complex)
        M[:N, :N] = TL
        M[N:, N:] = BR
        M[:N, N:] = TR0
        M[N:, :N] = BL0

        self.M = M
        self._lam0_current = 0.0 + 0.0j

        ii = np.arange(N)
        self._diag_TR = (ii, N + ii)      # diag of TR block
        self._diag_BL = (N + ii, ii)      # diag of BL block

    def set_lam0(self, lam0: complex):
        lam0 = complex(lam0)
        dlam = lam0 - self._lam0_current
        if dlam == 0.0:
            return
        # TR: (L - lam0 I) -> subtract dlam from diagonal
        self.M[self._diag_TR] -= dlam
        # BL: (L - lam0 I)† = (L† - conj(lam0) I) -> subtract conj(dlam)
        self.M[self._diag_BL] -= np.conj(dlam)
        self._lam0_current = lam0

    def index_at(self, lam0: complex, zero_tol: float) -> int:
        self.set_lam0(lam0)
        return localizer_index_ldl(self.M, zero_tol=zero_tol)


def sweep_index_adaptive_on_window_fast(
    L_mat: np.ndarray,
    X: np.ndarray,
    x0: float,
    re_min: float, re_max: float,
    im_min: float, im_max: float,
    *,
    kappa: float,
    zero_tol: float,
    n_coarse: int,
    n_refine: int,
    refine_nonzero: bool = True,
    refine_edges: bool = True,
):
    """
    Coarse grid on (re,im), then refine only 'interesting' tiles.
    Returns IDX_hi on an upsampled grid, plus the hi-res re/im axes.
    """
    pre = PanelLocalizerPrecomp(L_mat, X, x0=x0, kappa=kappa)

    re_c = np.linspace(re_min, re_max, n_coarse)
    im_c = np.linspace(im_min, im_max, n_coarse)

    # COARSE
    IDX_c = np.zeros((n_coarse, n_coarse), dtype=int)
    for yi, im0 in enumerate(im_c):
        for xi, re0 in enumerate(re_c):
            IDX_c[yi, xi] = pre.index_at(re0 + 1j * im0, zero_tol=zero_tol)

    # UPSAMPLE BACKGROUND
    n_hi = (n_coarse - 1) * n_refine + 1
    re_hi = np.linspace(re_min, re_max, n_hi)
    im_hi = np.linspace(im_min, im_max, n_hi)

    IDX_bg = np.repeat(np.repeat(IDX_c[:-1, :-1], n_refine, axis=0), n_refine, axis=1)
    IDX_hi = np.pad(IDX_bg, ((0, 1), (0, 1)), mode="edge")

    def cell_is_interesting(yi, xi):
        corners = np.array([
            IDX_c[yi, xi],
            IDX_c[yi, xi + 1],
            IDX_c[yi + 1, xi],
            IDX_c[yi + 1, xi + 1],
        ])
        if refine_nonzero and np.any(corners != 0):
            return True
        if refine_edges and (corners.max() != corners.min()):
            return True
        return False

    interesting = [
        (yi, xi)
        for yi in range(n_coarse - 1)
        for xi in range(n_coarse - 1)
        if cell_is_interesting(yi, xi)
    ]

    # REFINE
    for yi, xi in interesting:
        re0, re1 = re_c[xi], re_c[xi + 1]
        im0, im1 = im_c[yi], im_c[yi + 1]

        re_f = np.linspace(re0, re1, n_refine + 1)
        im_f = np.linspace(im0, im1, n_refine + 1)

        patch = np.zeros((n_refine + 1, n_refine + 1), dtype=int)
        for fj, imv in enumerate(im_f):
            for fi, rev in enumerate(re_f):
                patch[fj, fi] = pre.index_at(rev + 1j * imv, zero_tol=zero_tol)

        y0 = yi * n_refine
        x0i = xi * n_refine
        IDX_hi[y0:y0 + n_refine + 1, x0i:x0i + n_refine + 1] = patch

    return IDX_hi, re_hi, im_hi



# Panel helpers
def spectrum_window(lam: np.ndarray, pad_frac: float = 0.15):
    re_min, re_max = lam.real.min(), lam.real.max()
    im_min, im_max = lam.imag.min(), lam.imag.max()
    pad_re = pad_frac * max(1e-12, (re_max - re_min))
    pad_im = pad_frac * max(1e-12, (im_max - im_min))
    return (re_min - pad_re, re_max + pad_re), (im_min - pad_im, im_max + pad_im)


def compute_one_panel(spec, L_mat, lam, X, *, kappa, zero_tol):
    x0 = float(spec["x0"])

    if spec.get("re") is None or spec.get("im") is None:
        (re_min, re_max), (im_min, im_max) = spectrum_window(lam, pad_frac=0.15)
    else:
        re_min, re_max = spec["re"]
        im_min, im_max = spec["im"]

    IDX, re_g, im_g = sweep_index_adaptive_on_window_fast(
        L_mat=L_mat,
        X=X,
        x0=x0,
        re_min=re_min, re_max=re_max,
        im_min=im_min, im_max=im_max,
        n_coarse=int(spec["n_coarse"]),
        n_refine=int(spec["n_refine"]),
        refine_nonzero=True,
        refine_edges=True,
        kappa=kappa,
        zero_tol=zero_tol,
    )

    uniq = sorted(set(map(int, np.unique(IDX))))
    return dict(
        spec=spec,
        IDX=IDX,
        re=re_g,
        im=im_g,
        lam=lam,
        window=(re_min, re_max, im_min, im_max),
        uniq=uniq,
    )


def main():
    # Model setup
    params = BTCParams(N_spins=10, omega=1.0)
    build_L = build_liouvillian_builder(params)
    _, K_rank_mat, _ = build_operator_space_coordinates(params)
    X = K_rank_mat

    kappa = 1.0
    zero_tol = 1e-8

    # Panel specs
    gL, gH = 0.75, 1.5
    panel_specs = [
        dict(label="(a)", gamma=gL, x0=1.0, re=(-0.20, 0.02), im=(-2.5, 2.5), n_coarse=50, n_refine=20),
        dict(label="(b)", gamma=gH, x0=1.0, re=(-0.8, 0.05),  im=(-2.5, 2.5), n_coarse=50, n_refine=10),
        dict(label="(c)", gamma=gL, x0=2.0, re=(-0.45, 0.02), im=(-4, 4),     n_coarse=50, n_refine=20),
        dict(label="(d)", gamma=gH, x0=2.0, re=(-1.2, 0.05),  im=(-4, 4),      n_coarse=50, n_refine=10),
    ]


    # Cache L + spectrum per gamma
    unique_gammas = sorted({float(spec["gamma"]) for spec in panel_specs})
    L_cache = {}
    print("Precomputing L_mat + eigvals per gamma...")
    for g in unique_gammas:
        L_mat = build_L(g)
        lam = np.linalg.eigvals(L_mat)
        L_cache[g] = (L_mat, lam)
    print("Done.\n")


    # Compute panels in parallel
    t0 = time.perf_counter()
    pbar = tqdm(
        total=len(panel_specs),
        desc="Panels (overall)",
        unit="panel",
        leave=True,
        dynamic_ncols=True,
        smoothing=0.05,
    )

    with tqdm_joblib_with_panel_eta(pbar, total_panels=len(panel_specs), t_start=t0):
        results = Parallel(
            n_jobs=-1,
            backend="loky",
            prefer="processes",
            batch_size=1,
        )(
            delayed(compute_one_panel)(
                spec,
                L_cache[float(spec["gamma"])][0],
                L_cache[float(spec["gamma"])][1],
                X,
                kappa=kappa,
                zero_tol=zero_tol,
            )
            for spec in panel_specs
        )

    all_idx_vals = set()
    for p in results:
        for v in p["uniq"]:
            all_idx_vals.add(int(v))

    # checkpoint
    payload = dict(
        run_tag=RUN_TAG,
        N_spins=params.N_spins,
        j=params.j,
        omega=params.omega,
        kappa=kappa,
        zero_tol=zero_tol,
        panel_specs=panel_specs,
        panels_done=len(results),
        panels=results,
        all_idx_vals=sorted(all_idx_vals),
    )
    save_checkpoint(CKPT_PATH, payload)

    elapsed = time.perf_counter() - t0
    print(f"\nComputed {len(results)} panels in {elapsed:.2f}s")
    print(f"Saved checkpoint: {CKPT_PATH}")


    # Plotting
    panels = [
        dict(spec=p["spec"], IDX=p["IDX"], re=p["re"], im=p["im"], lam=p["lam"], window=p["window"])
        for p in results
    ]

    idx_vals_sorted = np.array(sorted(all_idx_vals), dtype=int)
    print("Index values appearing across all panels:", idx_vals_sorted)

    idx_vals = idx_vals_sorted.tolist()
    val_to_slot = {v: i for i, v in enumerate(idx_vals)}
    m = len(idx_vals)

    def remap_ID(IDX: np.ndarray) -> np.ndarray:
        out = np.empty_like(IDX, dtype=int)
        for v, s in val_to_slot.items():
            out[IDX == v] = s
        return out

    palette = [
        (0.94, 0.94, 0.94, 1.0),  # light gray
        (0.15, 0.42, 0.38, 1.0),  # teal
        (0.20, 0.20, 0.20, 1.0),  # charcoal
        (0.55, 0.27, 0.27, 1.0),  # brick
        (0.36, 0.28, 0.50, 1.0),  # purple
        (0.45, 0.45, 0.20, 1.0),  # olive
    ]
    if m > len(palette):
        tmp = plt.get_cmap("cividis", m)
        colors = [tmp(i) for i in range(m)]
    else:
        colors = palette[:m]
    if 0 in val_to_slot:
        colors[val_to_slot[0]] = (0.94, 0.94, 0.94, 1.0)

    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, m + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    contour_levels = np.arange(0.5, m, 1.0)

    fig = plt.figure(figsize=(12.8, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.075], wspace=0.06, hspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])

    for ax, p in zip(axes, panels):
        spec = p["spec"]
        IDX_plot = remap_ID(p["IDX"])

        re_g, im_g = p["re"], p["im"]
        lam = p["lam"]
        re_min, re_max, im_min, im_max = p["window"]

        ax.imshow(
            IDX_plot,
            origin="lower",
            aspect="auto",
            extent=[re_g.min(), re_g.max(), im_g.min(), im_g.max()],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
            zorder=0,
        )

        ax.axvline(0.0, color=(0, 0, 0, 0.18), lw=0.7, zorder=1)
        ax.axhline(0.0, color=(0, 0, 0, 0.12), lw=0.7, zorder=1)

        RR, II = np.meshgrid(re_g, im_g)
        ax.contour(
            RR, II, IDX_plot,
            levels=contour_levels,
            colors=[(0.0, 0.0, 0.0, 0.35)],
            linewidths=0.6,
            antialiased=True,
            zorder=2,
        )

        mask = (
            (lam.real >= re_min) & (lam.real <= re_max) &
            (lam.imag >= im_min) & (lam.imag <= im_max)
        )
        ax.scatter(
            lam.real[mask], lam.imag[mask],
            s=10, alpha=0.8,
            facecolors="none",
            edgecolors=(0.05, 0.05, 0.05, 0.55),
            linewidths=0.85,
            clip_on=True,
            rasterized=True,
            zorder=3,
        )

        header = rf"$\tilde\Gamma={spec['gamma']/params.omega:.2f}$" + r"\quad" + rf"$x_0={spec['x0']:.0f}$"
        ax.set_title(header, fontsize=14, pad=6, loc="center")

        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Im($\lambda_0$)", fontsize=15)
    for ax in axes:
        ax.set_xlabel(r"Re($\lambda_0$)", fontsize=15)

    # fixed limits (yours)
    axes[0].set_xlim(-0.20, 0.0 + 1e-2); axes[0].set_ylim(-2.5, 2.5)
    axes[1].set_xlim(-0.8, 0.05);        axes[1].set_ylim(-2.5, 2.5)
    axes[2].set_xlim(-0.45, 0.0 + 1e-2); axes[2].set_ylim(-4, 4)
    axes[3].set_xlim(-1.2, 0.05);        axes[3].set_ylim(-4, 4)

    for ax in axes:
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(m))
    cbar.ax.set_yticklabels([str(v) for v in idx_vals])
    cbar.set_label(r"$\nu^L$", fontsize=15)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=14)

    plt.show()

    out_pdf = f"index_1x4_v4_fast_{params.N_spins}.pdf"
    print(f"Saved figure: {out_pdf}")
    print(f"Saved checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()
# %%