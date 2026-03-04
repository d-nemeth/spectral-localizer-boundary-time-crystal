from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.linalg as sla


# LDL inertia utilities

def inertia_from_ldl_D(D: np.ndarray, tol: float = 1e-10) -> tuple[int, int, int]:
    """
    SciPy's ldl returns H = L @ D @ L^T.
    D is block diagonal with 1x1 and 2x2 pivot blocks.
    We compute the inertia (n_pos, n_neg, n_zero) by reading the 1x1 blocks
    directly and doing eigvalsh only on each 2x2 block.
    """
    n = D.shape[0]
    pos = neg = zero = 0
    i = 0
    eps2 = 1e-14  # threshold to detect a 2x2 pivot block via off-diagonal coupling

    while i < n:
        # Detect 2x2 pivot block
        if i < n - 1 and (abs(D[i + 1, i]) > eps2 or abs(D[i, i + 1]) > eps2):
            blk = D[i : i + 2, i : i + 2]
            ev = np.linalg.eigvalsh(blk).real  # cheap for 2x2
            pos += int(np.sum(ev > tol))
            neg += int(np.sum(ev < -tol))
            zero += int(2 - (np.sum(ev > tol) + np.sum(ev < -tol)))
            i += 2
        else:
            d = float(np.real(D[i, i]))
            if d > tol:
                pos += 1
            elif d < -tol:
                neg += 1
            else:
                zero += 1
            i += 1

    return pos, neg, zero


def localizer_index_ldl(L_loc: np.ndarray, zero_tol: float = 1e-10) -> int:
    """
    Compute the spectral localizer index via inertia (signature) using LDL.
      idx = - (pos - neg)//2
    The leading minus matches the convention from the standard implementation.
    """
    _, D, _ = sla.ldl(L_loc, hermitian=True)
    pos, neg, _ = inertia_from_ldl_D(D, tol=zero_tol)
    return -int((pos - neg) // 2)



# Precomputation class (cheap x0 updates)

class LocalizerPrecomp:
    """
    Build and store the 2N x 2N Hermitian localizer in block form:

      L_loc(x0) = [[  kappa (X - x0 I),     A ],
                  [       A^†        , -kappa (X - x0 I) ]]

    where A = L - lam0 I.

    Then sweeping x0 only changes the diagonals of the TL and BR blocks:
      TL diag -= kappa * Δx0
      BR diag += kappa * Δx0

    This turns each x0 step into O(N) diagonal update + LDL factorization.
    """
    def __init__(self, L_mat: np.ndarray, X: np.ndarray, lam0: complex, kappa: float, *, verbose: bool = False):
        self.N = int(L_mat.shape[0])
        self.kappa = float(kappa)

        N = self.N
        I = np.eye(N, dtype=complex)

        # A = L - lam0 I
        A = L_mat.astype(complex, copy=False) - lam0 * I
        Ad = A.conj().T

        # Hermitian safety for X
        Xh = 0.5 * (X + X.conj().T)
        Xk = self.kappa * Xh.astype(complex, copy=False)

        if verbose:
            print(f"||A|| = {np.linalg.norm(A):.3e}, ||kappa*X|| = {np.linalg.norm(Xk):.3e}")

        # Base localizer at x0 = 0
        L0 = np.empty((2 * N, 2 * N), dtype=complex)
        L0[:N, :N] = Xk
        L0[:N, N:] = A
        L0[N:, :N] = Ad
        L0[N:, N:] = -Xk

        self._L_work = L0
        self._x0_current = 0.0

        # Precompute diagonal index arrays for in-place updates
        ii = np.arange(N)
        self._tl = (ii, ii)          # TL diag indices
        self._br = (N + ii, N + ii)  # BR diag indices

    def set_x0(self, x0: float) -> None:
        x0 = float(x0)
        dx = x0 - self._x0_current
        if dx == 0.0:
            return

        shift = self.kappa * dx
        # TL: kappa(X - x0 I) => subtract shift on diagonal
        self._L_work[self._tl] -= shift
        # BR: -kappa(X - x0 I) => add shift on diagonal
        self._L_work[self._br] += shift

        self._x0_current = x0

    @property
    def matrix(self) -> np.ndarray:
        return self._L_work


def idx_at_x0(pre: LocalizerPrecomp, x0: float, zero_tol: float) -> int:
    pre.set_x0(x0)
    return localizer_index_ldl(pre.matrix, zero_tol=zero_tol)


# Adaptive 1D sweep in x0

def adaptive_index_sweep(
    L_mat: np.ndarray,
    X: np.ndarray,
    lam0: complex,
    *,
    x_min: float,
    x_max: float,
    kappa: float,
    zero_tol: float,
    n_coarse: int = 60,
    max_refine: int = 10,
    refine_only_changes: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute index nu^L(x0) on a nonuniform grid by:
      1) coarse uniform grid
      2) repeatedly insert midpoints only in intervals where the index changes

    Returns:
      x_sorted: (M,) array
      idx_sorted: (M,) array of ints
    """
    pre = LocalizerPrecomp(L_mat, X, lam0=lam0, kappa=kappa, verbose=verbose)

    x = np.linspace(float(x_min), float(x_max), int(n_coarse))
    idx = np.empty_like(x, dtype=int)

    # Evaluate coarse grid in increasing x for best reuse of diagonal updates
    pre.set_x0(x[0])
    idx[0] = localizer_index_ldl(pre.matrix, zero_tol=zero_tol)
    for i in range(1, len(x)):
        idx[i] = idx_at_x0(pre, x[i], zero_tol=zero_tol)

    # Refinement loop
    for _ in range(int(max_refine)):
        change = np.where(idx[:-1] != idx[1:])[0]
        if change.size == 0:
            break

        if refine_only_changes:
            mids = 0.5 * (x[change] + x[change + 1])
        else:
            mids = 0.5 * (x[:-1] + x[1:])

        mids = np.unique(mids)
        mids.sort()

        mid_idx = np.empty_like(mids, dtype=int)
        pre.set_x0(mids[0])
        mid_idx[0] = localizer_index_ldl(pre.matrix, zero_tol=zero_tol)
        for i in range(1, len(mids)):
            mid_idx[i] = idx_at_x0(pre, mids[i], zero_tol=zero_tol)

        x = np.concatenate([x, mids])
        idx = np.concatenate([idx, mid_idx])
        order = np.argsort(x)
        x = x[order]
        idx = idx[order]

    return x, idx



# Integration with btc_model.py

@dataclass(frozen=True)
class FastLocalizerConfig:
    kappa: float = 1.0
    zero_tol: float = 1e-10
    n_coarse: int = 60
    max_refine: int = 10
    refine_only_changes: bool = True
    verbose: bool = False


def compute_idx_curve_for_gamma(
    gamma: float,
    build_L: Callable[[float], np.ndarray],
    X: np.ndarray,
    lam0: complex,
    *,
    x_min: float,
    x_max: float,
    cfg: FastLocalizerConfig = FastLocalizerConfig(),
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Worker: build Liouvillian for this gamma, run adaptive sweep, return (gamma, x, idx).

    build_L is the output of build_liouvillian_builder(params) from btc_model.py:
      build_L(gamma) -> L_mat
    """
    L_mat = build_L(float(gamma))
    x, idx = adaptive_index_sweep(
        L_mat, X, lam0,
        x_min=x_min, x_max=x_max,
        kappa=cfg.kappa,
        zero_tol=cfg.zero_tol,
        n_coarse=cfg.n_coarse,
        max_refine=cfg.max_refine,
        refine_only_changes=cfg.refine_only_changes,
        verbose=cfg.verbose,
    )
    return float(gamma), x, idx