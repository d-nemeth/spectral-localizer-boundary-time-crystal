# %%
from __future__ import annotations

import os
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

from tqdm.auto import tqdm

from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    spectral_localizer,
    localizer_gap_and_index,
)

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

# model setup
params = BTCParams(N_spins=10, omega=1.0)
omega = params.omega
build_L = build_liouvillian_builder(params)
K2_mat, K_rank_mat, Q_mat = build_operator_space_coordinates(params)

X = K_rank_mat
coord_label = r"$K_{\mathrm{rank}}$"

kappa = 1.0
zero_tol = 1e-8

# checkpoint config
CHECKPOINT_DIR = Path("checkpoints_localizer")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_PATH = CHECKPOINT_DIR / f"localizer_panels_{RUN_TAG}.pkl"

def save_checkpoint(path: Path, payload: dict):
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# adaptive sweep
def sweep_index_adaptive_on_window(
    L_mat: np.ndarray,
    x0: float,
    re_min: float, re_max: float,
    im_min: float, im_max: float,
    n_coarse: int = 30,
    n_refine: int = 8,
    refine_nonzero: bool = True,
    refine_edges: bool = True,
    pbar_desc: str = "",
):
    re_c = np.linspace(re_min, re_max, n_coarse)
    im_c = np.linspace(im_min, im_max, n_coarse)

    IDX_c = np.zeros((n_coarse, n_coarse), dtype=int)
    total_coarse = n_coarse * n_coarse

    t0 = time.perf_counter()
    pbar1 = tqdm(total=total_coarse, desc=f"{pbar_desc} coarse", leave=False)

    cnt = 0
    for yi, im0 in enumerate(im_c):
        for xi, re0 in enumerate(re_c):
            lam0 = re0 + 1j * im0
            L_loc = spectral_localizer(L_mat, X, lam0=lam0, x0=x0, kappa=kappa)
            _, IDX_c[yi, xi] = localizer_gap_and_index(L_loc, zero_tol=zero_tol)

            cnt += 1
            if (cnt % max(1, total_coarse // 200)) == 0:
                elapsed = time.perf_counter() - t0
                rate = cnt / max(elapsed, 1e-12)
                eta = (total_coarse - cnt) / max(rate, 1e-12)
                pbar1.set_postfix_str(f"ETA {eta/60:.1f} min")
            pbar1.update(1)

    pbar1.close()

    # upsample background
    n_hi  = (n_coarse - 1) * n_refine + 1
    re_hi = np.linspace(re_min, re_max, n_hi)
    im_hi = np.linspace(im_min, im_max, n_hi)

    IDX_bg = np.repeat(np.repeat(IDX_c[:-1, :-1], n_refine, axis=0), n_refine, axis=1)
    IDX_hi = np.pad(IDX_bg, ((0, 1), (0, 1)), mode="edge")

    def cell_is_interesting(yi, xi):
        corners = np.array([
            IDX_c[yi,   xi],
            IDX_c[yi,   xi+1],
            IDX_c[yi+1, xi],
            IDX_c[yi+1, xi+1],
        ])
        if refine_nonzero and np.any(corners != 0):
            return True
        if refine_edges and (corners.max() != corners.min()):
            return True
        return False

    interesting = [(yi, xi) for yi in range(n_coarse - 1) for xi in range(n_coarse - 1)
                   if cell_is_interesting(yi, xi)]

    # refine only interesting tiles
    t1 = time.perf_counter()
    pbar2 = tqdm(total=len(interesting), desc=f"{pbar_desc} refine tiles", leave=False)

    done = 0
    for yi, xi in interesting:
        re0, re1 = re_c[xi], re_c[xi+1]
        im0, im1 = im_c[yi], im_c[yi+1]

        re_f = np.linspace(re0, re1, n_refine + 1)
        im_f = np.linspace(im0, im1, n_refine + 1)

        patch = np.zeros((n_refine + 1, n_refine + 1), dtype=int)
        for fj, imv in enumerate(im_f):
            for fi, rev in enumerate(re_f):
                lam0 = rev + 1j * imv
                L_loc = spectral_localizer(L_mat, X, lam0=lam0, x0=x0, kappa=kappa)
                _, patch[fj, fi] = localizer_gap_and_index(L_loc, zero_tol=zero_tol)

        y0 = yi * n_refine
        x0i = xi * n_refine
        IDX_hi[y0:y0 + n_refine + 1, x0i:x0i + n_refine + 1] = patch

        done += 1
        if len(interesting) > 0 and (done % max(1, len(interesting) // 100)) == 0:
            elapsed = time.perf_counter() - t1
            rate = done / max(elapsed, 1e-12)
            eta = (len(interesting) - done) / max(rate, 1e-12)
            pbar2.set_postfix_str(f"ETA {eta/60:.1f} min")
        pbar2.update(1)

    pbar2.close()
    return IDX_hi, re_hi, im_hi

# panel specs
gL, gH = 0.75, 1.5
panel_specs = [
    dict(label="(a)", gamma=gL, x0=1.0, re=(-0.20, 0.02), im=(-2.5, 2.5), n_coarse=20, n_refine=5),
    dict(label="(b)", gamma=gH, x0=1.0, re=(-0.8, 0.05),  im=(-2.5, 2.5), n_coarse=20,  n_refine=5),
    dict(label="(c)", gamma=gL, x0=2.0, re=(-0.45, 0.02), im=(-4, 4),     n_coarse=20, n_refine=10),
    dict(label="(d)", gamma=gH, x0=2.0, re=(-1.2, 0.05),  im=(-4, 4),      n_coarse=20,  n_refine=5),
]

def spectrum_window(lam: np.ndarray, pad_frac: float = 0.15):
    re_min, re_max = lam.real.min(), lam.real.max()
    im_min, im_max = lam.imag.min(), lam.imag.max()
    pad_re = pad_frac * max(1e-12, (re_max - re_min))
    pad_im = pad_frac * max(1e-12, (im_max - im_min))
    return (re_min - pad_re, re_max + pad_re), (im_min - pad_im, im_max + pad_im)

# compute panels
panels = []
all_idx_vals = set()

panel_pbar = tqdm(panel_specs, desc="Panels (overall)", leave=True)
t_panels = time.perf_counter()

for spec in panel_pbar:
    g = spec["gamma"]
    x0 = spec["x0"]
    panel_pbar.set_postfix_str(f"{spec['label']} Γ={g:.2f} x0={x0:.0f}")

    L_mat = build_L(g)
    lam = np.linalg.eigvals(L_mat)

    re_min, re_max = spec["re"]
    im_min, im_max = spec["im"]

    IDX, re_g, im_g = sweep_index_adaptive_on_window(
        L_mat=L_mat,
        x0=x0,
        re_min=re_min, re_max=re_max,
        im_min=im_min, im_max=im_max,
        n_coarse=spec["n_coarse"],
        n_refine=spec["n_refine"],
        refine_nonzero=True,
        refine_edges=True,
        pbar_desc=f"{spec['label']} Γ={g:.2f} x0={x0:.0f}",
    )

    for v in np.unique(IDX):
        all_idx_vals.add(int(v))

    panels.append(dict(
        spec=spec,
        IDX=IDX,
        re=re_g,
        im=im_g,
        lam=lam,
        window=(re_min, re_max, im_min, im_max),
    ))

    payload = {
        "run_tag": RUN_TAG,
        "N_spins": params.N_spins,
        "j": params.j,
        "omega": params.omega,
        "kappa": kappa,
        "zero_tol": zero_tol,
        "panel_specs": panel_specs,
        "panels_done": len(panels),
        "panels": panels,
        "all_idx_vals": sorted(all_idx_vals),
    }
    save_checkpoint(CKPT_PATH, payload)

panel_pbar.close()

idx_vals_sorted = np.array(sorted(all_idx_vals), dtype=int)
print("Index values appearing across all panels:", idx_vals_sorted)


# REMAP indices so colormap/norm only represent values that appear
idx_vals = idx_vals_sorted.tolist()
val_to_slot = {v: i for i, v in enumerate(idx_vals)}
m = len(idx_vals)

def remap_ID(IDX: np.ndarray) -> np.ndarray:
    out = np.empty_like(IDX, dtype=int)
    for v, s in val_to_slot.items():
        out[IDX == v] = s
    return out

palette = [
    (0.94, 0.94, 0.94, 1.0),  # light gray (for index 0)
    (0.15, 0.42, 0.38, 1.0),  # deep teal
    (0.20, 0.20, 0.20, 1.0),  # charcoal
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

# Plot: 1x4 side-by-side + shared colorbar 
fig = plt.figure(figsize=(12.8, 3.2), constrained_layout=True)
gs  = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.075], wspace=0.06, hspace=0.05)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
]
cax = fig.add_subplot(gs[0, 4])

for ax, p in zip(axes, panels):
    spec = p["spec"]
    IDX  = p["IDX"]
    IDX_plot = remap_ID(IDX)

    re_g, im_g = p["re"], p["im"]
    lam = p["lam"]
    re_min, re_max, im_min, im_max = p["window"]

    imh = ax.imshow(
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
        zorder=2
    )

    mask = (
        (lam.real >= re_min) & (lam.real <= re_max) &
        (lam.imag >= im_min) & (lam.imag <= im_max)
    )
    ax.scatter(
        lam.real[mask], lam.imag[mask], s=25, alpha=0.8,
        facecolors="none",
        edgecolors=(0.05, 0.05, 0.05, 0.55),
        linewidths=0.85,
        clip_on=True,
        rasterized=True,
        zorder=3,
    )

    header = (
        rf"$\tilde\Gamma={spec['gamma']/omega:.2f}$"
        r"\quad"
        rf"$x_0={spec['x0']:.0f}$"
    )
    ax.set_title(header, fontsize=14, pad=6, loc="center")

    ax.tick_params(labelsize=8)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(r"Im($\lambda_0$)", fontsize=15)
for ax in axes[1:]:
    ax.set_ylabel("")

for ax in axes:
    ax.set_xlabel(r"Re($\lambda_0$)", fontsize=15)

# KEEP fixed limits
axes[0].set_xlim(-0.20, 0.0 + 1e-2); axes[0].set_ylim(-2.5, 2.5)
axes[1].set_xlim(-0.8, 0.05);        axes[1].set_ylim(-2.5, 2.5)
axes[2].set_xlim(-0.45, 0.0 + 1e-2); axes[2].set_ylim(-4, 4)
axes[3].set_xlim(-1.2, 0.05);        axes[3].set_ylim(-4, 4)

for ax in axes:
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
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
#fig.savefig(f"index_1x4_v4_{N_spins}.pdf", bbox_inches="tight")
print(f"Saved figure: index_1x4_v4_{params.N_spins}.pdf")
print(f"Saved checkpoint: {CKPT_PATH}")
# %%