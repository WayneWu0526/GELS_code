"""
MaPS Estimator: Implements Algorithm 1 from the paper.
6-DoF pose estimation from magnetic measurements.
"""
import numpy as np
from kabsch_solver import kabsch_solver


def null(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Null space of A (MATLAB null equivalent). Returns (n, k) orthonormal basis."""
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    n = A.shape[1] if A.ndim > 1 else len(A)
    rank = int(np.sum(s > rcond * s[0])) if s.size > 0 else 0
    k = n - rank
    return Vh[rank:, :].T if k > 0 else np.zeros((n, 0))


def dipole_field_matrix(p: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Build B(p; p0) such that b = B(p; p0) @ m (point-dipole)."""
    r_vec = np.asarray(p, dtype=float).flatten() - np.asarray(p0, dtype=float).flatten()
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("Zero source-observer distance.")
    r_hat = r_vec / r
    coeff = 4.0 * np.pi * 1e-7 / (4.0 * np.pi)
    return coeff * (3.0 * np.outer(r_hat, r_hat) - np.eye(3)) / r**3


def compute_rho_hat_eigen(X: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
    """
    Estimate rho_hat from gradient tensor X and local field b_hat.

    Uses eigenvalue decomposition to avoid direct inversion:
        rho_hat = 3 / (lambda_max * lambda_min) * (X + lambda_med * I) @ b_hat
    """
    evals = np.sort(np.linalg.eigvalsh(X))[::-1]   # descending
    lam_max, lam_med, lam_min = evals[0], evals[1], evals[2]
    return 3.0 / (lam_max * lam_min) * (X + lam_med * np.eye(3)) @ b_hat


def compute_local_quantities(D_cal: np.ndarray, B_meas_cell: list) -> dict:
    """
    Estimate local magnetic fields and gradient tensors for each source slot.

    Args:
        D_cal:       (3, N) sensor offset matrix in sensor frame (meters)
        B_meas_cell: list of M matrices, each (3, N), in Tesla

    Returns:
        dict with keys: b_hat_locals (3, M), X_hat_locals (list of M 3x3), w, Pw
    """
    D_cal = np.asarray(D_cal, dtype=float)
    if D_cal.ndim != 2 or D_cal.shape[0] != 3:
        raise ValueError(f"D_cal must be (3, N), got {D_cal.shape}")

    N = D_cal.shape[1]
    M = len(B_meas_cell)
    tol = 1e-10
    ones_N = np.ones((N, 1))

    # Local field weight vector w such that b_hat = B_meas @ w
    Q_bar = null(D_cal)
    if Q_bar.size == 0:
        if np.linalg.norm(D_cal @ ones_N) < tol:
            w = ones_N / N
        else:
            raise ValueError("Cannot build w: null(D_cal) empty and array not centered.")
    else:
        g = Q_bar.T @ ones_N
        g_norm_sq = (g.T @ g).item()
        if g_norm_sq < tol:
            raise ValueError("Invalid geometry: 1 has zero projection onto null(D_cal).")
        w = (Q_bar @ g) / g_norm_sq

    Pw = np.eye(N) - w @ ones_N.T

    # Constraint matrix for symmetric traceless gradient tensor (5 independent params)
    S_mat = np.array([
        [1,  0,  0,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  1,  0,  0,  0],
        [0,  0,  0,  1,  0],
        [0,  0,  0,  0,  1],
        [0,  0,  1,  0,  0],
        [0,  0,  0,  0,  1],
        [-1, 0,  0, -1,  0],
    ], dtype=float)

    C_mat = np.kron(D_cal.T, np.eye(3)) @ S_mat
    if np.linalg.matrix_rank(C_mat, tol=tol) < 5:
        raise ValueError("Gradient tensor not uniquely identifiable: rank(C_mat) < 5. "
                         "Need at least 3 non-degenerate sensors.")
    C_pinv = np.linalg.pinv(C_mat)

    b_hat_locals = np.zeros((3, M))
    X_hat_locals = []

    for i in range(M):
        B_meas = np.asarray(B_meas_cell[i], dtype=float)
        if B_meas.shape != (3, N):
            raise ValueError(f"B_meas_cell[{i}]: expected (3, {N}), got {B_meas.shape}")
        b_hat = B_meas @ w
        b_hat_locals[:, i] = b_hat.ravel()

        B_tilde = B_meas @ Pw
        h_vec = B_tilde.reshape(-1, order='F')
        x_hat = C_pinv @ h_vec
        X_hat = (S_mat @ x_hat).reshape((3, 3), order='F')
        X_hat_locals.append(X_hat)

    return {'b_hat_locals': b_hat_locals, 'X_hat_locals': X_hat_locals, 'w': w, 'Pw': Pw}


def MaPS_Estimator(D_cal: np.ndarray, sources: list, B_meas_cell: list):
    """
    MaPS Estimator: Algorithm 1 — 6-DoF pose estimation.

    Args:
        D_cal:       (3, N) sensor offsets in sensor frame (meters)
        sources:     list of M dicts, each with 'p_Ci' (3,) and 'm_Ci' (3,)
        B_meas_cell: list of M (3, N) magnetic measurement matrices (Tesla)

    Returns:
        R_est:   (3, 3) estimated rotation matrix
        p_est:   (3,)  estimated position (meters)
        details: dict with intermediate values (b_hat_locals, X_hat_locals, rho_hats, ...)
    """
    M = len(B_meas_cell)
    local = compute_local_quantities(D_cal, B_meas_cell)
    b_hat_locals = local['b_hat_locals']
    X_hat_locals = local['X_hat_locals']

    rho_hats = np.zeros((3, M))
    for i in range(M):
        rho_hats[:, i] = compute_rho_hat_eigen(X_hat_locals[i], b_hat_locals[:, i])

    # Global pose recovery via centered Kabsch
    # p_Ck = p - R @ rho_k  =>  centered: p_C_tilde = -R @ rho_tilde
    p_Ck = np.column_stack([sources[i]['p_Ci'] for i in range(M)])
    p_C_bar = np.mean(p_Ck, axis=1, keepdims=True)
    rho_bar = np.mean(rho_hats, axis=1, keepdims=True)
    p_C_tilde = p_Ck - p_C_bar
    rho_tilde = rho_hats - rho_bar

    R_est = kabsch_solver(p_C_tilde, -rho_tilde)
    p_est = (p_C_bar + R_est @ rho_bar).reshape(3)

    details = {
        'b_hat_locals': b_hat_locals,
        'X_hat_locals': X_hat_locals,
        'rho_hats': rho_hats,
        'rho_bar': rho_bar,
        'p_C_bar': p_C_bar,
    }
    return R_est, p_est, details
