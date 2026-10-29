#!/usr/bin/env python3
"""Evaluate localization in a rectangular workspace at five fake-MoveIt states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR
SENSOR_PACKAGE_ROOT = THIS_DIR / "vendor"
if str(SENSOR_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SENSOR_PACKAGE_ROOT))

from sensor_array_config import get_config  # noqa: E402

from simulation_core import localize_case, sample_target_poses  # noqa: E402
from source_geometry import compute_g_shape  # noqa: E402


def workspace_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return "packaged working snapshot; see docs/provenance.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sources(state: dict, magnitude: float, sign: float) -> list[dict]:
    rows = []
    for arm_name, target in state["targets"].items():
        axis = np.asarray(target["axis_direction"], dtype=float)
        axis /= np.linalg.norm(axis)
        rows.append(
            {
                "source_index": int(target["vertex_index"]),
                "arm_name": arm_name,
                "position_m": np.asarray(target["position_m"], dtype=float),
                "axis_direction": axis,
                "moment_A_m2": sign * magnitude * axis,
                "tilt_from_down_deg": float(target["tilt_from_down_deg"]),
            }
        )
    return sorted(rows, key=lambda row: row["source_index"])


def run(config_path: Path, output_override: Path | None = None) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    experiment = config["experiment"]
    source_config = config["sources"]
    target_config = config["targets"]
    measurement_config = config["measurement"]
    trajectory_path = workspace_path(experiment["trajectory_results"])
    with trajectory_path.open("r", encoding="utf-8") as handle:
        trajectory = json.load(handle)
    if trajectory.get("verification_status") != "MOVEIT_FAKE_ONLY":
        raise ValueError("trajectory must come from the guarded fake-MoveIt search")
    if trajectory.get("execute_called") is not False:
        raise ValueError("trajectory record unexpectedly reports execution")
    winner = trajectory.get("winner")
    if not winner or len(winner.get("states", [])) != 5:
        raise ValueError("trajectory must contain five successful representative states")

    output_path = output_override or workspace_path(config["output"]["directory"])
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Use a fresh output directory: " + str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    sensor_config = get_config("QMC6309")
    sensor_offsets = np.asarray(sensor_config.hardware.d_list, dtype=float).T
    if sensor_offsets.shape != (3, 12):
        raise ValueError(f"expected the 12-sensor array, received {sensor_offsets.shape}")

    candidate = winner["candidate"]
    normal = np.asarray(candidate["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    cube_center = np.asarray(candidate["centroid_m"], dtype=float) + float(
        target_config["center_offset_along_frustum_normal_m"]
    ) * normal
    half_extent = np.asarray(target_config["half_extent_m"], dtype=float)
    if cube_center[2] - half_extent[2] < float(target_config["minimum_z_m"]):
        raise ValueError("configured rectangular workspace extends below the table surface")
    target_poses = sample_target_poses(
        count=int(target_config["count"]),
        center_m=cube_center,
        half_extent_m=half_extent,
        seed=int(experiment["random_seed"]),
        orientation_mode=str(target_config["orientation_mode"]),
    )

    target_rows = [
        {
            "target_index": pose["target_index"],
            "x_m": pose["position_m"][0],
            "y_m": pose["position_m"][1],
            "z_m": pose["position_m"][2],
            "qx": pose["quaternion_xyzw"][0],
            "qy": pose["quaternion_xyzw"][1],
            "qz": pose["quaternion_xyzw"][2],
            "qw": pose["quaternion_xyzw"][3],
        }
        for pose in target_poses
    ]
    source_rows = []
    error_rows = []
    target_summary_rows = []
    summary_rows = []
    magnitude = float(source_config["moment_magnitude_A_m2"])
    sign = float(source_config["moment_axis_sign"])
    trials = int(measurement_config["trials_per_target"])
    noise_std = float(measurement_config["noise_std_T"])

    for state_index, state in enumerate(winner["states"]):
        g_shape = float(state["g_shape"])
        sources = ordered_sources(state, magnitude, sign)
        positions = np.stack([row["position_m"] for row in sources])
        moments = np.stack([row["moment_A_m2"] for row in sources])
        actual_g = compute_g_shape(positions)
        for row in sources:
            source_rows.append(
                {
                    "state_index": state_index,
                    "g_shape_requested": g_shape,
                    "g_shape_actual": actual_g,
                    "source_index": row["source_index"],
                    "arm_name": row["arm_name"],
                    "x_m": row["position_m"][0],
                    "y_m": row["position_m"][1],
                    "z_m": row["position_m"][2],
                    "axis_x": row["axis_direction"][0],
                    "axis_y": row["axis_direction"][1],
                    "axis_z": row["axis_direction"][2],
                    "moment_x_A_m2": row["moment_A_m2"][0],
                    "moment_y_A_m2": row["moment_A_m2"][1],
                    "moment_z_A_m2": row["moment_A_m2"][2],
                    "tilt_from_down_deg": row["tilt_from_down_deg"],
                }
            )

        noise_rng = np.random.default_rng(int(experiment["noise_seed"]))
        position_errors = {pose["target_index"]: [] for pose in target_poses}
        orientation_errors = {pose["target_index"]: [] for pose in target_poses}
        success_count = 0
        for pose in target_poses:
            for trial_index in range(trials):
                noise = noise_rng.normal(
                    0.0, noise_std, size=(3, 3, sensor_offsets.shape[1])
                )
                record = {
                    "state_index": state_index,
                    "g_shape": actual_g,
                    "target_index": pose["target_index"],
                    "trial_index": trial_index,
                    "success": False,
                    "position_error_mm": "",
                    "orientation_error_deg": "",
                    "error_type": "",
                }
                try:
                    result = localize_case(
                        sensor_offsets_m=sensor_offsets,
                        source_positions_m=positions,
                        source_moments_A_m2=moments,
                        target_pose=pose,
                        model_order=int(source_config["model_order"]),
                        additive_noise_T=noise,
                    )
                    record.update(
                        {
                            "success": True,
                            "position_error_mm": result["position_error_mm"],
                            "orientation_error_deg": result["orientation_error_deg"],
                        }
                    )
                    success_count += 1
                    position_errors[pose["target_index"]].append(
                        result["position_error_mm"]
                    )
                    orientation_errors[pose["target_index"]].append(
                        result["orientation_error_deg"]
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record["error_type"] = type(error).__name__
                error_rows.append(record)

        target_position_means = []
        target_orientation_means = []
        for pose in target_poses:
            index = pose["target_index"]
            position_values = position_errors[index]
            orientation_values = orientation_errors[index]
            position_mean = float(np.mean(position_values)) if position_values else ""
            orientation_mean = (
                float(np.mean(orientation_values)) if orientation_values else ""
            )
            if position_values:
                target_position_means.append(position_mean)
                target_orientation_means.append(orientation_mean)
            target_summary_rows.append(
                {
                    "state_index": state_index,
                    "g_shape": actual_g,
                    "target_index": index,
                    "trial_count": trials,
                    "success_count": len(position_values),
                    "success_rate": len(position_values) / trials,
                    "position_error_mean_mm": position_mean,
                    "orientation_error_mean_deg": orientation_mean,
                }
            )

        total_count = len(target_poses) * trials
        summary_rows.append(
            {
                "state_index": state_index,
                "g_shape": actual_g,
                "total_count": total_count,
                "success_count": success_count,
                "success_rate": success_count / total_count,
                "position_error_mean_mm": np.mean(target_position_means),
                "position_error_median_mm": np.median(target_position_means),
                "position_error_p10_mm": np.percentile(target_position_means, 10),
                "position_error_p90_mm": np.percentile(target_position_means, 90),
                "orientation_error_mean_deg": np.mean(target_orientation_means),
                "diana7_tilt_from_down_deg": state["targets"]["diana7"][
                    "tilt_from_down_deg"
                ],
                "surface_clearance_mm": state["surface_clearance_mm"],
            }
        )
        print(f"g={actual_g:.6f}: {success_count}/{total_count} successful", flush=True)

    write_csv(output_path / "target_poses.csv", list(target_rows[0]), target_rows)
    write_csv(output_path / "source_states.csv", list(source_rows[0]), source_rows)
    write_csv(output_path / "localization_errors.csv", list(error_rows[0]), error_rows)
    write_csv(
        output_path / "target_error_summary.csv",
        list(target_summary_rows[0]),
        target_summary_rows,
    )
    write_csv(output_path / "summary_by_state.csv", list(summary_rows[0]), summary_rows)
    with (output_path / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "UNVERIFIED_ASSUMED_NOISE",
        "label": experiment["label"],
        "git_head": git_head(),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256(trajectory_path),
        "trajectory_verification_status": trajectory["verification_status"],
        "trajectory_recorded": trajectory.get("trajectory_recorded", False),
        "trajectory_executed": trajectory.get("trajectory_executed", False),
        "source_moment_provenance": source_config["moment_provenance"],
        "cube_center_m": cube_center.tolist(),
        "cube_half_extent_m": half_extent.tolist(),
        "target_count": len(target_poses),
        "trials_per_target": trials,
        "noise_std_T": noise_std,
        "noise_provenance": measurement_config["noise_provenance"],
    }
    with (output_path / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"results: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
