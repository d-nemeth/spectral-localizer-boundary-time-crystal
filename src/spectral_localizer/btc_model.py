from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import qutip as qt


@dataclass(frozen=True)
class BTCParams:
    """Model parameters for the collective-spin BTC model."""
    N_spins: int = 10
    omega: float = 1.0

    @property
    def j(self) -> float:
        return self.N_spins / 2.0

    @property
    def d(self) -> int:
        return int(2 * self.j + 1)


def build_spin_operators(j: float):
    """Return collective spin operators Jx, Jy, Jz, Jm."""
    Jx = qt.jmat(j, "x")
    Jy = qt.jmat(j, "y")
    Jz = qt.jmat(j, "z")
    Jm = qt.jmat(j, "-")
    return Jx, Jy, Jz, Jm


def build_adjoint_generators(Jx, Jy, Jz):
    """Adjoint (Liouville-space) generators K_alpha = spre(J_alpha) - spost(J_alpha)."""
    Kx = qt.spre(Jx) - qt.spost(Jx)
    Ky = qt.spre(Jy) - qt.spost(Jy)
    Kz = qt.spre(Jz) - qt.spost(Jz)
    return Kx, Ky, Kz


def build_liouvillian_builder(params: BTCParams):
    """
    Return a function build_liouvillian(gamma) -> dense np.ndarray.

    Keeps Jx, Jm, N_spins captured in closure, so you don't recreate operators
    every time in a sweep.
    """
    j = params.j
    N_spins = params.N_spins
    omega = params.omega

    Jx, Jy, Jz, Jm = build_spin_operators(j)

    def build_liouvillian(gamma: float) -> np.ndarray:
        H = omega * Jx
        c = np.sqrt(float(gamma) / float(N_spins)) * Jm
        L = qt.liouvillian(H, c_ops=[c])
        return L.full()

    return build_liouvillian


def build_rank_operator_from_casimir(K2_mat: np.ndarray, j: float) -> np.ndarray:
    """
    Build K_rank by diagonalizing K^2 and mapping eigenvalues to k(k+1).

    Returns Hermitian matrix K_rank_mat (same basis as K2_mat).
    """
    K2h = 0.5 * (K2_mat + K2_mat.conj().T)
    evals, U = np.linalg.eigh(K2h)
    evals = evals.real

    k_max = int(round(2 * j))
    k_values = np.arange(k_max + 1, dtype=float)
    casimir_vals = k_values * (k_values + 1)

    k_labels = np.zeros_like(evals)
    for i, ev in enumerate(evals):
        idx = int(np.argmin(np.abs(ev - casimir_vals)))
        k_labels[i] = k_values[idx]

    K_rank_diag = np.diag(k_labels)
    K_rank_mat = U @ K_rank_diag @ U.conj().T
    K_rank_mat = 0.5 * (K_rank_mat + K_rank_mat.conj().T)
    return K_rank_mat


def build_operator_space_coordinates(params: BTCParams):
    """
    Build and return (K2_mat, K_rank_mat, Q_mat) for the model.
    """
    j = params.j
    Jx, Jy, Jz, Jm = build_spin_operators(j)
    Kx, Ky, Kz = build_adjoint_generators(Jx, Jy, Jz)

    K2_mat = (Kx * Kx + Ky * Ky + Kz * Kz).full()
    K2_mat = 0.5 * (K2_mat + K2_mat.conj().T)

    K_rank_mat = build_rank_operator_from_casimir(K2_mat, j)

    Q_mat = Kz.full()
    Q_mat = 0.5 * (Q_mat + Q_mat.conj().T)

    # Hermiticity checks
    if not np.allclose(K_rank_mat, K_rank_mat.conj().T, atol=1e-8):
        raise ValueError("K_rank_mat not Hermitian to tolerance.")
    if not np.allclose(Q_mat, Q_mat.conj().T, atol=1e-8):
        raise ValueError("Q_mat not Hermitian to tolerance.")

    return K2_mat, K_rank_mat, Q_mat