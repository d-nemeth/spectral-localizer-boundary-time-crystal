from __future__ import annotations
import numpy as np


def spectral_localizer(
    L_mat: np.ndarray,
    X: np.ndarray,
    lam0: complex,
    x0: float,
    kappa: float,
    hermitian_atol: float = 1e-8,
) -> np.ndarray:
    """
    Standard spectral localizer using Pauli-matrix kron form.

    A = L - lam0 I
    A1 = (A + A†)/2
    A2 = -(i/2)(A - A†)
    A3 = kappa (X - x0 I)

    L_loc = kron(A1, sigma_x) + kron(A2, sigma_y) + kron(A3, sigma_z)
    """
    N = int(L_mat.shape[0])
    I = np.eye(N, dtype=complex)

    A = L_mat.astype(complex, copy=False) - lam0 * I
    A1 = 0.5 * (A + A.conj().T)
    A2 = -0.5j * (A - A.conj().T)
    A3 = float(kappa) * (X.astype(complex, copy=False) - float(x0) * I)

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    L_loc = np.kron(A1, sx) + np.kron(A2, sy) + np.kron(A3, sz)

    if not np.allclose(L_loc, L_loc.conj().T, atol=hermitian_atol):
        raise ValueError("Localizer is not Hermitian. Check X and inputs.")
    return L_loc


def localizer_gap_and_index(L_loc: np.ndarray, zero_tol: float = 1e-8) -> tuple[float, int]:
    """
    Compute the localizer gap mu = min|eig| and index = signature/2.
    Uses full eigvalsh (reference method).
    """
    evals = np.linalg.eigvalsh(L_loc)
    mu = float(np.min(np.abs(evals)))

    pos = int(np.sum(evals > zero_tol))
    neg = int(np.sum(evals < -zero_tol))
    sig = pos - neg
    idx = int(sig // 2)
    return mu, idx