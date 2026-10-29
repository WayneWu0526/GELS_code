#!/usr/bin/env python3
"""Run a reproducible source-geometry sweep for the GELS estimator."""

from __future__ import annotations

import argparse
import csv
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
from source_geometry import generate_source_positions, geometry_summary  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return "packaged working snapshot; see docs/provenance.md"


def run(config_path: Path, output_override: Path | None = None) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_path = output_override or Path(config["output"]["directory"])
    if not output_path.is_absolute():
        output_path = WORKSPACE_ROOT / output_path
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Use a fresh output directory: " + str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    sensor_config = get_config("QMC6309")
    sensor_offsets = np.asarray(sensor_config.hardware.d_list, dtype=float).T
    if sensor_offsets.shape != (3, 12):
        raise ValueError(f"expected the 12-sensor array, received {sensor_offsets.shape}")

    geometry_config = config["geometry"]
    source_config = config["sources"]
    target_config = config["targets"]
    measurement_config = config["measurement"]
    experiment_config = config["experiment"]

    target_poses = sample_target_poses(
        count=int(target_config["count"]),
        center_m=target_config["center_m"],
        half_extent_m=target_config["half_extent_m"],
        seed=int(experiment_config["random_seed"]),
        orientation_mode=str(target_config["orientation_mode"]),
    )
    moment = np.asarray(source_config["moment_A_m2"], dtype=float)
    source_moments = np.repeat(moment[None, :], 3, axis=0)
    trials = int(measurement_config["trials_per_target"])
    noise_std = float(measurement_config["noise_std_T"])

    target_rows = []
    for pose in target_poses:
        target_rows.append(
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
        )

    source_rows = []
    error_rows = []
    target_summary_rows = []
    summary_rows = []
    for g_value in [float(value) for value in geometry_config["g_values"]]:
        positions = generate_source_positions(
            g_shape=g_value,
            centroid_m=geometry_config["centroid_m"],
            frobenius_scale_m=float(geometry_config["frobenius_scale_m"]),
            plane_basis=geometry_config["plane_basis"],
        )
        geometry = geometry_summary(positions)
        for source_index, position in enumerate(positions):
            source_rows.append(
                {
                    "g_shape_requested": g_value,
                    "g_shape_actual": geometry["g_shape"],
                    "source_index": source_index,
                    "x_m": position[0],
                    "y_m": position[1],
                    "z_m": position[2],
                    "frobenius_scale_m": geometry["frobenius_scale_m"],
                }
            )

        noise_rng = np.random.default_rng(int(experiment_config["noise_seed"]))
        position_errors_by_target = {
            pose["target_index"]: [] for pose in target_poses
        }
        orientation_errors_by_target = {
            pose["target_index"]: [] for pose in target_poses
        }
        success_count = 0
        for pose in target_poses:
            for trial_index in range(trials):
                noise = noise_rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=(3, 3, sensor_offsets.shape[1]),
                )
                row = {
                    "g_shape": geometry["g_shape"],
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
                        source_moments_A_m2=source_moments,
                        target_pose=pose,
                        model_order=int(source_config["model_order"]),
                        additive_noise_T=noise,
                    )
                    row.update(
                        {
                            "success": True,
                            "position_error_mm": result["position_error_mm"],
                            "orientation_error_deg": result["orientation_error_deg"],
                        }
                    )
                    success_count += 1
                    position_errors_by_target[pose["target_index"]].append(
                        result["position_error_mm"]
                    )
                    orientation_errors_by_target[pose["target_index"]].append(
                        result["orientation_error_deg"]
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row["error_type"] = type(error).__name__
                error_rows.append(row)

        total_count = len(target_poses) * trials
        target_position_means = []
        target_orientation_means = []
        for pose in target_poses:
            target_index = pose["target_index"]
            position_values = position_errors_by_target[target_index]
            orientation_values = orientation_errors_by_target[target_index]
            target_success_count = len(position_values)
            position_mean = np.mean(position_values) if position_values else ""
            orientation_mean = np.mean(orientation_values) if orientation_values else ""
            if position_values:
                target_position_means.append(float(position_mean))
                target_orientation_means.append(float(orientation_mean))
            target_summary_rows.append(
                {
                    "g_shape": geometry["g_shape"],
                    "target_index": target_index,
                    "trial_count": trials,
                    "success_count": target_success_count,
                    "success_rate": target_success_count / trials,
                    "position_error_mean_mm": position_mean,
                    "orientation_error_mean_deg": orientation_mean,
                }
            )
        summary_rows.append(
            {
                "g_shape": geometry["g_shape"],
                "total_count": total_count,
                "success_count": success_count,
                "success_rate": success_count / total_count,
                "position_error_mean_mm": np.mean(target_position_means) if target_position_means else "",
                "position_error_median_mm": np.median(target_position_means) if target_position_means else "",
                "position_error_p10_mm": np.percentile(target_position_means, 10) if target_position_means else "",
                "position_error_p90_mm": np.percentile(target_position_means, 90) if target_position_means else "",
                "orientation_error_mean_deg": np.mean(target_orientation_means) if target_orientation_means else "",
                "orientation_error_median_deg": np.median(target_orientation_means) if target_orientation_means else "",
                "orientation_error_p10_deg": np.percentile(target_orientation_means, 10) if target_orientation_means else "",
                "orientation_error_p90_deg": np.percentile(target_orientation_means, 90) if target_orientation_means else "",
            }
        )
        print(
            f"g={geometry['g_shape']:.6f} complete: "
            f"{success_count}/{total_count} successful",
            flush=True,
        )

    write_csv(output_path / "target_poses.csv", list(target_rows[0]), target_rows)
    write_csv(output_path / "source_configurations.csv", list(source_rows[0]), source_rows)
    write_csv(output_path / "localization_errors.csv", list(error_rows[0]), error_rows)
    write_csv(
        output_path / "target_error_summary.csv",
        list(target_summary_rows[0]),
        target_summary_rows,
    )
    write_csv(output_path / "summary_by_gshape.csv", list(summary_rows[0]), summary_rows)
    with (output_path / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "UNVERIFIED",
        "label": experiment_config["label"],
        "git_head": git_head(),
        "config_path": str(config_path.resolve()),
        "sensor_count": int(sensor_offsets.shape[1]),
        "g_count": len(summary_rows),
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
