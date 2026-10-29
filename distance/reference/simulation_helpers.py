#!/usr/bin/env python3
"""Frozen numerical helpers used by the distance sweep."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
GELS_SCRIPTS_DIR = THIS_DIR.parent / "vendor"
GSHAPE_DIR = THIS_DIR
SENSOR_PACKAGE_ROOT = THIS_DIR.parent / "vendor"
for search_path in (GELS_SCRIPTS_DIR, GSHAPE_DIR, SENSOR_PACKAGE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from maps_estimator import MaPS_Estimator  # noqa: E402
from sensor_array_config import get_config  # noqa: E402
from simulation_core import (  # noqa: E402
    quaternion_xyzw_to_rotation,
    rotation_error_deg,
    synthesize_measurements,
)




def load_sources(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["source_index"]))
    if [int(row["source_index"]) for row in rows] != [0, 1, 2]:
        raise ValueError("source CSV must contain source indices 0, 1, and 2")
    positions = np.array(
        [[float(row[axis]) for axis in ("x_m", "y_m", "z_m")] for row in rows]
    )
    directions = np.array(
        [[float(row[axis]) for axis in ("axis_x", "axis_y", "axis_z")] for row in rows]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return positions, directions


def load_targets(path: Path) -> list[dict]:
    targets = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            quaternion = np.array(
                [float(row[axis]) for axis in ("qx", "qy", "qz", "qw")]
            )
            targets.append(
                {
                    "target_index": int(row["target_index"]),
                    "position_m": np.array(
                        [float(row[axis]) for axis in ("x_m", "y_m", "z_m")]
                    ),
                    "rotation": quaternion_xyzw_to_rotation(quaternion),
                }
            )
    if not targets:
        raise ValueError("target CSV is empty")
    return targets


def constraint_matrix(sensor_offsets_m: np.ndarray) -> np.ndarray:
    s_matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [-1, 0, 0, -1, 0],
        ],
        dtype=float,
    )
    return np.kron(sensor_offsets_m.T, np.eye(3)) @ s_matrix


def affine_weights(sensor_offsets_m: np.ndarray) -> np.ndarray:
    system = np.vstack([sensor_offsets_m, np.ones((1, sensor_offsets_m.shape[1]))])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    weights, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    if np.linalg.norm(system @ weights - target) > 1.0e-10:
        raise ValueError("selected sensors cannot reconstruct the field at the array origin")
    return weights


def base_field_norms_G(
    sensor_offsets_m: np.ndarray,
    source_positions_m: np.ndarray,
    source_directions: np.ndarray,
    targets: list[dict],
    model_order: int,
) -> np.ndarray:
    zero_noise = np.zeros((3, 3, sensor_offsets_m.shape[1]), dtype=float)
    norms = []
    for target in targets:
        fields = synthesize_measurements(
            sensor_offsets_m=sensor_offsets_m,
            source_positions_m=source_positions_m,
            source_moments_A_m2=source_directions,
            target_position_m=target["position_m"],
            target_rotation=target["rotation"],
            model_order=model_order,
            additive_noise_T=zero_noise,
        )
        norms.extend(
            (np.linalg.norm(np.asarray(fields), axis=1) * 1.0e4).reshape(-1).tolist()
        )
    return np.asarray(norms, dtype=float)

