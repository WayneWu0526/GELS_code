"""ROS-independent magnetic measurement and target-pose utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_GELS_SCRIPTS_DIR = _THIS_DIR.parent / "vendor"
if str(_GELS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_GELS_SCRIPTS_DIR))

from mag_dipole_model import mag_dipole_model  # noqa: E402
from maps_estimator import MaPS_Estimator  # noqa: E402


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def sample_target_poses(
    count: int,
    center_m: np.ndarray,
    half_extent_m: np.ndarray,
    seed: int,
    orientation_mode: str = "random",
) -> list[dict]:
    """Generate a deterministic Latin-hypercube position set."""
    if count <= 0:
        raise ValueError("count must be positive")
    center = np.asarray(center_m, dtype=float).reshape(3)
    half_extent = np.asarray(half_extent_m, dtype=float).reshape(3)
    if np.any(half_extent < 0.0):
        raise ValueError("half_extent_m cannot be negative")

    rng = np.random.default_rng(seed)
    unit_samples = np.empty((count, 3), dtype=float)
    for dimension in range(3):
        unit_samples[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    positions = center + (2.0 * unit_samples - 1.0) * half_extent

    poses = []
    for index, position in enumerate(positions):
        if orientation_mode == "identity":
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        elif orientation_mode == "random":
            u1, u2, u3 = rng.random(3)
            quaternion = np.array(
                [
                    np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
                    np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
                    np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
                    np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
                ]
            )
        else:
            raise ValueError(f"unsupported orientation_mode: {orientation_mode}")
        poses.append(
            {
                "target_index": index,
                "position_m": position,
                "quaternion_xyzw": quaternion,
                "rotation": quaternion_xyzw_to_rotation(quaternion),
            }
        )
    return poses


def synthesize_measurements(
    sensor_offsets_m: np.ndarray,
    source_positions_m: np.ndarray,
    source_moments_A_m2: np.ndarray,
    target_position_m: np.ndarray,
    target_rotation: np.ndarray,
    model_order: int,
    additive_noise_T: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return one 3xN sensor-frame field matrix per source."""
    offsets = np.asarray(sensor_offsets_m, dtype=float)
    positions = np.asarray(source_positions_m, dtype=float)
    moments = np.asarray(source_moments_A_m2, dtype=float)
    target_position = np.asarray(target_position_m, dtype=float).reshape(3)
    rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)

    if offsets.ndim != 2 or offsets.shape[0] != 3:
        raise ValueError("sensor_offsets_m must have shape (3, N)")
    if positions.shape != (3, 3) or moments.shape != (3, 3):
        raise ValueError("source positions and moments must have shape (3, 3)")

    sensor_positions_global = target_position[:, None] + rotation @ offsets
    field_matrices = []
    for source_index in range(3):
        measurement = np.empty_like(offsets)
        for sensor_index in range(offsets.shape[1]):
            field_global, _ = mag_dipole_model(
                sensor_positions_global[:, sensor_index],
                moments[source_index],
                positions[source_index],
                order=model_order,
            )
            measurement[:, sensor_index] = rotation.T @ field_global
        field_matrices.append(measurement)

    if additive_noise_T is not None:
        noise = np.asarray(additive_noise_T, dtype=float)
        if noise.shape != (3, 3, offsets.shape[1]):
            raise ValueError("additive_noise_T must have shape (3, 3, N)")
        field_matrices = [
            field_matrices[index] + noise[index] for index in range(3)
        ]
    return field_matrices


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    relative = np.asarray(estimate).T @ np.asarray(truth)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def localize_case(
    sensor_offsets_m: np.ndarray,
    source_positions_m: np.ndarray,
    source_moments_A_m2: np.ndarray,
    target_pose: dict,
    model_order: int,
    additive_noise_T: np.ndarray | None,
) -> dict:
    fields = synthesize_measurements(
        sensor_offsets_m=sensor_offsets_m,
        source_positions_m=source_positions_m,
        source_moments_A_m2=source_moments_A_m2,
        target_position_m=target_pose["position_m"],
        target_rotation=target_pose["rotation"],
        model_order=model_order,
        additive_noise_T=additive_noise_T,
    )
    sources = [
        {"p_Ci": source_positions_m[index], "m_Ci": source_moments_A_m2[index]}
        for index in range(3)
    ]
    rotation_estimate, position_estimate, details = MaPS_Estimator(
        sensor_offsets_m,
        sources,
        fields,
    )
    return {
        "position_estimate_m": position_estimate,
        "rotation_estimate": rotation_estimate,
        "position_error_mm": float(
            np.linalg.norm(position_estimate - target_pose["position_m"]) * 1000.0
        ),
        "orientation_error_deg": rotation_error_deg(
            rotation_estimate,
            target_pose["rotation"],
        ),
        "details": details,
    }
