from __future__ import annotations
import numpy as np

def build_kq_basis_from_casimir_and_Q(
    K2_mat: np.ndarray,
    Q_mat: np.ndarray,
    j: float,
) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray]:
    K2 = 0.5 * (K2_mat + K2_mat.conj().T)
    Q  = 0.5 * (Q_mat  + Q_mat.conj().T)

    evals, U = np.linalg.eigh(K2)
    evals = evals.real

    k_max = int(round(2 * j))
    k_values = np.arange(k_max + 1, dtype=float)
    casimir_vals = k_values * (k_values + 1)

    k_labels = np.zeros_like(evals)
    for i, ev in enumerate(evals):
        idx = int(np.argmin(np.abs(ev - casimir_vals)))
        k_labels[i] = k_values[idx]

    k_values_present = np.unique(k_labels).astype(int)
    k_values_present.sort()

    basis_vectors: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []

    for k in k_values_present:
        idxs = np.where(k_labels == k)[0]
        V = U[:, idxs]

        Qk = V.conj().T @ Q @ V
        q_evals, W = np.linalg.eigh(Qk)
        q_evals = q_evals.real

        Bk = V @ W

        q_targets = np.arange(-k, k + 1, dtype=int)
        q_labels = [int(q_targets[np.argmin(np.abs(ev - q_targets))]) for ev in q_evals]

        order = np.argsort(q_labels)
        for t in order:
            basis_vectors.append(Bk[:, t])
            labels.append((int(k), int(q_labels[t])))

    B = np.column_stack(basis_vectors)
    err = np.linalg.norm(B.conj().T @ B - np.eye(B.shape[1]))
    print(f"(k,q) basis built. Orthonormality ||B†B-I|| = {err:.2e}")
    print("k sectors present:", k_values_present)

    return basis_vectors, labels, k_values_present
