"""
mag_dipole_model: Magnetic dipole field and gradient tensor model.
"""
import numpy as np


def mag_dipole_model(p: np.ndarray, m: np.ndarray, pC: np.ndarray,
                     order: int = 1, L: float = 240.0e-3, D: float = 40.0e-3):
    """
    Computes the magnetic field and gradient tensor for a magnetic dipole.

    Args:
        p:     Observation position (3,) in meters
        m:     Dipole moment vector (3,) in A·m²
        pC:    Dipole center position (3,) in meters
        order: Model order — 1 (point dipole), 3 (+ octupole), 5 (+ hexapole)
        L:     Cylinder length in meters (used for order >= 3 corrections)
        D:     Cylinder diameter in meters (used for order >= 3 corrections)

    Returns:
        b: Magnetic field vector (3,) in Tesla
        A: Magnetic gradient tensor (3x3) in T/m
    """
    if order not in [1, 3, 5]:
        raise ValueError("order must be 1, 3, or 5")

    p = np.asarray(p, dtype=float).flatten()
    m = np.asarray(m, dtype=float).flatten()
    pC = np.asarray(pC, dtype=float).flatten()

    mu0 = 4 * np.pi * 1e-7
    coeff = mu0 / (4 * np.pi)

    r_vec = p - pC
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("Observation point coincides with dipole center.")
    r_hat = r_vec / r

    b1 = coeff * (3 * np.dot(m, r_hat) * r_hat - m) / r**3
    b = b1.copy()

    if order >= 3:
        beta = D / L
        m_hat = m / (np.linalg.norm(m) + 1e-10)
        m_dot_r = np.dot(m_hat, r_hat)
        m_dot_r_sq = m_dot_r**2

        coef_b3 = coeff * (1 / r**5) * (L / 2)**2 * ((4 - 3 * beta**2) / 8)
        term1 = (35 * m_dot_r_sq - 15) * np.outer(r_hat, r_hat)
        term2 = (15 * m_dot_r_sq - 3) * np.eye(3)
        b3 = coef_b3 * ((term1 - term2) @ m)
        b = b1 + b3

    if order >= 5:
        beta = D / L
        m_hat = m / (np.linalg.norm(m) + 1e-10)
        m_dot_r = np.dot(m_hat, r_hat)
        m_dot_r_sq = m_dot_r**2
        m_dot_r_4 = m_dot_r_sq**2

        coef_b5 = coeff * (1 / r**7) * (L / 2)**4 * ((15 * beta**4 - 60 * beta**2 + 24) / 64)
        coeff_p = 231 * m_dot_r_4 - (105 / 2) * m_dot_r_sq + 35
        coeff_I = 105 * m_dot_r_4 - 70 * m_dot_r_sq + 5
        b5 = coef_b5 * ((coeff_p * np.outer(r_hat, r_hat) - coeff_I * np.eye(3)) @ m)
        b = b1 + b3 + b5

    # Gradient tensor (order 1 only)
    m_dot_r_vec = np.dot(m, r_vec)
    A = coeff * (3 / r**5) * (
        m_dot_r_vec * np.eye(3)
        + np.outer(r_vec, m)
        + np.outer(m, r_vec)
        - (5 * m_dot_r_vec / r**2) * np.outer(r_vec, r_vec)
    )

    return b, A


def eul2rotm(euler_zyx_rad: np.ndarray) -> np.ndarray:
    """
    ZYX Euler angles (radians) to rotation matrix.

    Args:
        euler_zyx_rad: [rz, ry, rx] in radians

    Returns:
        R: (3, 3) rotation matrix  R = Rz @ Ry @ Rx
    """
    rz, ry, rx = euler_zyx_rad
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [ 0,          1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0,          0          ],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    return Rz @ Ry @ Rx


def rotm2eul_zyx(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix to ZYX Euler angles (radians).

    Returns:
        [rz, ry, rx] in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rz, ry, rx])
