"""Controlled three-source geometry for the GELS conditioning study."""

from __future__ import annotations

import numpy as np


_SOURCE_BASIS = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
    ],
    dtype=float,
)


def compute_g_shape(source_positions: np.ndarray, tolerance: float = 1.0e-14) -> float:
    """Return sigma_2 / sigma_1 for three source positions stored by row."""
    positions = np.asarray(source_positions, dtype=float)
    if positions.shape != (3, 3):
        raise ValueError("source_positions must have shape (3, 3)")

    centered = (positions - positions.mean(axis=0)).T
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] <= tolerance:
        raise ValueError("source geometry has zero scale")
    return float(singular_values[1] / singular_values[0])


def generate_source_positions(
    g_shape: float,
    centroid_m: np.ndarray,
    frobenius_scale_m: float,
    plane_basis: np.ndarray | None = None,
) -> np.ndarray:
    """Generate three positions with prescribed shape and fixed total scale.

    The returned array has shape ``(3, 3)`` and stores one source position per
    row. At ``g_shape=0`` the sources are distinct and collinear. At
    ``g_shape=1`` they form an equilateral triangle.
    """
    g_value = float(g_shape)
    if not 0.0 <= g_value <= 1.0:
        raise ValueError("g_shape must be in [0, 1]")

    scale = float(frobenius_scale_m)
    if scale <= 0.0:
        raise ValueError("frobenius_scale_m must be positive")

    centroid = np.asarray(centroid_m, dtype=float).reshape(-1)
    if centroid.shape != (3,):
        raise ValueError("centroid_m must contain three values")

    if plane_basis is None:
        plane = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    else:
        plane = np.asarray(plane_basis, dtype=float)
    if plane.shape != (3, 2):
        raise ValueError("plane_basis must have shape (3, 2)")
    if not np.allclose(plane.T @ plane, np.eye(2), atol=1.0e-10):
        raise ValueError("plane_basis columns must be orthonormal")

    sigma_1 = scale / np.sqrt(1.0 + g_value * g_value)
    sigma_2 = g_value * sigma_1
    centered_columns = plane @ np.diag([sigma_1, sigma_2]) @ _SOURCE_BASIS
    return centered_columns.T + centroid


def geometry_summary(source_positions: np.ndarray) -> dict:
    """Return invariant quantities used by tests and result manifests."""
    positions = np.asarray(source_positions, dtype=float)
    if positions.shape != (3, 3):
        raise ValueError("source_positions must have shape (3, 3)")
    centered = positions - positions.mean(axis=0)
    sides = np.array(
        [
            np.linalg.norm(positions[0] - positions[1]),
            np.linalg.norm(positions[1] - positions[2]),
            np.linalg.norm(positions[2] - positions[0]),
        ]
    )
    return {
        "g_shape": compute_g_shape(positions),
        "centroid_m": positions.mean(axis=0),
        "frobenius_scale_m": float(np.linalg.norm(centered)),
        "side_lengths_m": sides,
    }
