from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from run_utils.run_manager import (
    get_next_run_dir,
)
from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
)
from spectral_localizer.fast_localizer import (
    localizer_index_ldl,
)

# Global parameters
N_SPINS = 10
OMEGA = 1.0

KAPPA = 1.0
ZERO_TOL = 1e-8

N_JOBS = 8

G_L = 0.5
G_H = 5.0

X_0 = 3.0


# Low-gamma panel window
L_RE_MIN = -0.5
L_RE_MAX = 0.01

L_IM_MIN = -5.0
L_IM_MAX = 5.0

# High-gamma panel window
H_RE_MIN = -12.0
H_RE_MAX = 0.5

H_IM_MIN = -5.0
H_IM_MAX = 5.0


# Adaptive resolutions
L_N_COARSE = 300
L_N_REFINE = 50

H_N_COARSE = 120
H_N_REFINE = 20


# Localizer precomputation
class PanelLocalizerPrecomp:
    def __init__(
        self,
        L_mat: np.ndarray,
        X: np.ndarray,
        x0: float,
        kappa: float,
    ):

        self.N = int(L_mat.shape[0])

        self.kappa = float(kappa)

        self.x0 = float(x0)

        N = self.N

        Xh = 0.5 * (X + X.conj().T)

        TL = self.kappa * (
            Xh
            - self.x0
            * np.eye(
                N,
                dtype=complex,
            )
        )

        BR = -TL

        TR0 = L_mat.astype(
            complex,
            copy=False,
        )

        BL0 = TR0.conj().T

        M = np.empty(
            (2 * N, 2 * N),
            dtype=complex,
        )

        M[:N, :N] = TL
        M[N:, N:] = BR
        M[:N, N:] = TR0
        M[N:, :N] = BL0

        self.M = M

        self._lam0_current = 0.0 + 0.0j

        ii = np.arange(N)

        self._diag_TR = (
            ii,
            N + ii,
        )

        self._diag_BL = (
            N + ii,
            ii,
        )

    def set_lam0(
        self,
        lam0: complex,
    ):

        lam0 = complex(lam0)

        dlam = lam0 - self._lam0_current

        if dlam == 0.0:
            return

        self.M[self._diag_TR] -= dlam

        self.M[self._diag_BL] -= np.conj(dlam)

        self._lam0_current = lam0

    def index_at(
        self,
        lam0: complex,
        zero_tol: float,
    ) -> int:

        self.set_lam0(lam0)

        return localizer_index_ldl(
            self.M,
            zero_tol=zero_tol,
        )


# Parallel coarse rows
def compute_coarse_row(
    yi: int,
    im0: float,
    re_c: np.ndarray,
    L_mat: np.ndarray,
    X: np.ndarray,
    x0: float,
    kappa: float,
    zero_tol: float,
):

    pre = PanelLocalizerPrecomp(
        L_mat=L_mat,
        X=X,
        x0=x0,
        kappa=kappa,
    )

    row = np.zeros(
        len(re_c),
        dtype=int,
    )

    for xi, re0 in enumerate(re_c):
        row[xi] = pre.index_at(
            re0 + 1j * im0,
            zero_tol=zero_tol,
        )

    return yi, row


# Parallel refined patches
def compute_refined_patch(
    yi: int,
    xi: int,
    re_c: np.ndarray,
    im_c: np.ndarray,
    n_refine: int,
    L_mat: np.ndarray,
    X: np.ndarray,
    x0: float,
    kappa: float,
    zero_tol: float,
):

    pre = PanelLocalizerPrecomp(
        L_mat=L_mat,
        X=X,
        x0=x0,
        kappa=kappa,
    )

    re0 = re_c[xi]
    re1 = re_c[xi + 1]

    im0 = im_c[yi]
    im1 = im_c[yi + 1]

    re_f = np.linspace(
        re0,
        re1,
        n_refine + 1,
    )

    im_f = np.linspace(
        im0,
        im1,
        n_refine + 1,
    )

    patch = np.zeros(
        (
            n_refine + 1,
            n_refine + 1,
        ),
        dtype=int,
    )

    for fj, imv in enumerate(im_f):
        for fi, rev in enumerate(re_f):
            patch[fj, fi] = pre.index_at(
                rev + 1j * imv,
                zero_tol=zero_tol,
            )

    return yi, xi, patch


# Adaptive scan
def sweep_index_adaptive_on_window_fast(
    L_mat: np.ndarray,
    X: np.ndarray,
    x0: float,
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
    *,
    kappa: float,
    zero_tol: float,
    n_coarse: int,
    n_refine: int,
    n_jobs: int,
    refine_nonzero: bool = True,
    refine_edges: bool = True,
):

    re_c = np.linspace(
        re_min,
        re_max,
        n_coarse,
    )

    im_c = np.linspace(
        im_min,
        im_max,
        n_coarse,
    )

    row_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
    )(
        delayed(compute_coarse_row)(
            yi,
            im0,
            re_c,
            L_mat,
            X,
            x0,
            kappa,
            zero_tol,
        )
        for yi, im0 in enumerate(
            tqdm(
                im_c,
                desc="Coarse rows",
                leave=False,
            )
        )
    )

    idx_c = np.zeros(
        (
            n_coarse,
            n_coarse,
        ),
        dtype=int,
    )

    for yi, row in row_results:
        idx_c[yi, :] = row

    n_hi = (n_coarse - 1) * n_refine + 1

    re_hi = np.linspace(
        re_min,
        re_max,
        n_hi,
    )

    im_hi = np.linspace(
        im_min,
        im_max,
        n_hi,
    )

    idx_bg = np.repeat(
        np.repeat(
            idx_c[:-1, :-1],
            n_refine,
            axis=0,
        ),
        n_refine,
        axis=1,
    )

    idx_hi = np.pad(
        idx_bg,
        ((0, 1), (0, 1)),
        mode="edge",
    )

    def cell_is_interesting(
        yi,
        xi,
    ):

        corners = np.array(
            [
                idx_c[yi, xi],
                idx_c[yi, xi + 1],
                idx_c[yi + 1, xi],
                idx_c[yi + 1, xi + 1],
            ]
        )

        if refine_nonzero and np.any(corners != 0):
            return True

        if refine_edges and (corners.max() != corners.min()):
            return True

        return False

    interesting = [
        (yi, xi)
        for yi in range(n_coarse - 1)
        for xi in range(n_coarse - 1)
        if cell_is_interesting(
            yi,
            xi,
        )
    ]

    print(f"Refining {len(interesting)} cells")

    patch_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
    )(
        delayed(compute_refined_patch)(
            yi,
            xi,
            re_c,
            im_c,
            n_refine,
            L_mat,
            X,
            x0,
            kappa,
            zero_tol,
        )
        for yi, xi in tqdm(
            interesting,
            desc="Refined cells",
            leave=False,
        )
    )

    for yi, xi, patch in patch_results:
        y0 = yi * n_refine
        x0i = xi * n_refine

        idx_hi[
            y0 : y0 + n_refine + 1,
            x0i : x0i + n_refine + 1,
        ] = patch

    return idx_hi, re_hi, im_hi


# Compute one panel
def compute_one_panel(
    spec,
    L_mat,
    eigvals,
    X,
    *,
    kappa,
    zero_tol,
    n_jobs,
):

    print(f"\nComputing panel {spec['label']} at Gamma = {spec['gamma']:.3f}")

    idx, re_g, im_g = sweep_index_adaptive_on_window_fast(
        L_mat=L_mat,
        X=X,
        x0=float(spec["x0"]),
        re_min=spec["re"][0],
        re_max=spec["re"][1],
        im_min=spec["im"][0],
        im_max=spec["im"][1],
        kappa=kappa,
        zero_tol=zero_tol,
        n_coarse=int(spec["n_coarse"]),
        n_refine=int(spec["n_refine"]),
        n_jobs=n_jobs,
    )

    return {
        "label": spec["label"],
        "gamma": float(spec["gamma"]),
        "x0": float(spec["x0"]),
        "re": re_g,
        "im": im_g,
        "idx": idx,
        "eigvals": eigvals,
    }


# Main
def main():

    t0 = time.perf_counter()

    base_dir = Path("main_figures_datasets") / "topological_islands"

    run_dir = get_next_run_dir(base_dir)

    run_name = run_dir.name

    figures_dir = run_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nCreating run: {run_name}\n")

    params = BTCParams(
        N_spins=N_SPINS,
        omega=OMEGA,
    )

    panel_specs = [
        dict(
            label="a",
            gamma=G_L,
            x0=X_0,
            re=(
                L_RE_MIN,
                L_RE_MAX,
            ),
            im=(
                L_IM_MIN,
                L_IM_MAX,
            ),
            n_coarse=L_N_COARSE,
            n_refine=L_N_REFINE,
        ),
        dict(
            label="b",
            gamma=G_H,
            x0=X_0,
            re=(
                H_RE_MIN,
                H_RE_MAX,
            ),
            im=(
                H_IM_MIN,
                H_IM_MAX,
            ),
            n_coarse=H_N_COARSE,
            n_refine=H_N_REFINE,
        ),
    ]

    build_L = build_liouvillian_builder(params)

    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    X = K_rank_mat

    unique_gammas = sorted({float(spec["gamma"]) for spec in panel_specs})

    L_cache = {}

    print("Precomputing Liouvillians...")

    for gamma in unique_gammas:
        L_mat = build_L(gamma)

        eigvals = np.linalg.eigvals(L_mat)

        L_cache[gamma] = (
            L_mat,
            eigvals,
        )

    print("Done.\n")

    results = []

    for spec in panel_specs:
        panel = compute_one_panel(
            spec,
            L_cache[float(spec["gamma"])][0],
            L_cache[float(spec["gamma"])][1],
            X,
            kappa=KAPPA,
            zero_tol=ZERO_TOL,
            n_jobs=N_JOBS,
        )

        results.append(panel)

    labels = []
    gammas = []
    x0_vals = []

    re_arrays = []
    im_arrays = []
    idx_arrays = []
    eigvals_arrays = []

    all_idx_vals = set()

    for panel in results:
        labels.append(panel["label"])

        gammas.append(panel["gamma"])

        x0_vals.append(panel["x0"])

        re_arrays.append(panel["re"])

        im_arrays.append(panel["im"])

        idx_arrays.append(panel["idx"])

        eigvals_arrays.append(panel["eigvals"])

        all_idx_vals |= set(np.unique(panel["idx"]).tolist())

    data_file = run_dir / "data.npz"

    np.savez_compressed(
        data_file,
        labels=np.array(
            labels,
            dtype=str,
        ),
        gammas=np.array(
            gammas,
            dtype=float,
        ),
        x0_vals=np.array(
            x0_vals,
            dtype=float,
        ),
        re_arrays=np.array(
            re_arrays,
            dtype=object,
        ),
        im_arrays=np.array(
            im_arrays,
            dtype=object,
        ),
        idx_arrays=np.array(
            idx_arrays,
            dtype=object,
        ),
        eigvals_arrays=np.array(
            eigvals_arrays,
            dtype=object,
        ),
        unique_idx_vals=np.array(
            sorted(all_idx_vals),
            dtype=int,
        ),
    )

    print(f"\nSaved dataset:\n{data_file}")

    elapsed = time.perf_counter() - t0

    metadata = {
        "run_name": run_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(elapsed),
        "params": asdict(params),
        "kappa": float(KAPPA),
        "zero_tol": float(ZERO_TOL),
        "n_jobs": int(N_JOBS),
        "x0": float(X_0),
        "gamma_low": float(G_L),
        "gamma_high": float(G_H),
        "low_panel_resolution": {
            "n_coarse": int(L_N_COARSE),
            "n_refine": int(L_N_REFINE),
        },
        "high_panel_resolution": {
            "n_coarse": int(H_N_COARSE),
            "n_refine": int(H_N_REFINE),
        },
        "panel_specs": panel_specs,
        "parallelized_over": "grid_rows_and_refined_cells",
        "description": ("Adaptive fast spectral localizer island scans."),
    }

    info_file = run_dir / "info.json"

    with open(
        info_file,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(f"Saved metadata:\n{info_file}")

    print(f"\nFinished {run_name} in {elapsed:.2f} s\n")


if __name__ == "__main__":
    main()
