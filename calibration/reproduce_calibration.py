#!/usr/bin/env python3
"""Reproduce the QMC6309 affine fit from the preserved CSV/config snapshot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT
CONFIG = ROOT / "config"
DATA = ROOT / "data"


def delta_o(samples):
    """Centered vector RMS across sensors, one value per recorded sample."""
    centered = samples - samples.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1))


def load_inputs():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for item in manifest["files"]:
        digest = hashlib.sha256((SNAPSHOT / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Snapshot changed: {item['path']}")

    hardware = json.loads((CONFIG / "sensor_array_params.json").read_text())
    positions = np.asarray(hardware["d_list"], dtype=float)
    rotations = np.empty((12, 3, 3))
    assigned = set()
    for entry in hardware["R_CORR"]:
        matrix = np.asarray(entry["matrix"], dtype=float).reshape(3, 3, order="F")
        for sid in entry["sensor_ids"]:
            rotations[sid - 1] = matrix
            assigned.add(sid)
    if assigned != set(range(1, 13)):
        raise ValueError("R_CORR must cover all 12 sensors")

    _, singular, vh = np.linalg.svd(positions.T, full_matrices=True)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    q = vh[rank:].T
    g = q.T @ np.ones(12)
    weights = q @ g / (g @ g)

    configurations = []
    expected_header = [f"sensor_{sid}_{axis}" for sid in range(1, 13) for axis in "xyz"]
    for channel in "xyz":
        for voltage in range(1, 6):
            path = DATA / f"manual_{channel}/manual_record_{voltage}V.csv"
            with path.open(newline="") as stream:
                if next(csv.reader(stream)) != expected_header:
                    raise ValueError(f"Unexpected columns: {path}")
            raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            if raw.shape[1] != 36 or not np.isfinite(raw).all():
                raise ValueError(f"Invalid readings: {path}")
            raw = raw.reshape(-1, 12, 3)
            aligned = np.einsum("sij,nsj->nsi", rotations, raw)
            reference = np.einsum("nsi,s->ni", aligned, weights)
            magnitude = np.linalg.norm(reference, axis=1)
            if np.any(magnitude == 0):
                raise ValueError(f"Zero reference magnitude: {path}")
            target = reference * (magnitude.mean() / magnitude[:, None])
            configurations.append({
                "channel": channel.upper(), "voltage_V": voltage,
                "raw": raw, "aligned": aligned, "reference": reference,
                "target": target, "mean_reference_G": float(magnitude.mean()),
            })
    return configurations, weights


def fit_parameters(configurations):
    aligned = np.concatenate([c["aligned"] for c in configurations])
    target = np.concatenate([c["target"] for c in configurations])
    matrices, offsets, ranks, rmse = [], [], [], []
    for sensor in range(12):
        design = np.column_stack((aligned[:, sensor], np.ones(len(aligned))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank != 4:
            raise ValueError(f"Sensor {sensor + 1}: augmented design rank is {rank}")
        matrices.append(coefficients[:3].T)
        offsets.append(coefficients[3])
        ranks.append(int(rank))
        rmse.append(float(np.sqrt(np.mean((design @ coefficients - target)**2))))
    return np.asarray(matrices), np.asarray(offsets), ranks, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    protected = [ROOT, CONFIG, DATA, ROOT / "reference_results"]
    if any(output == p or (p != ROOT and p in output.parents) for p in protected):
        raise ValueError("Choose an output directory outside the preserved inputs and reference results")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory: " + str(output))
    configurations, weights = load_inputs()
    matrices, offsets, ranks, rmse = fit_parameters(configurations)
    stored = json.loads((CONFIG / "affine_model_params.json").read_text())
    stored_matrices = np.asarray([stored[str(i)]["D_i"] for i in range(1, 13)])
    stored_offsets = np.asarray([stored[str(i)]["e_i"] for i in range(1, 13)])
    rows, sample_rows, all_pre, all_post = [], [], [], []
    for cfg in configurations:
        aligned = cfg["aligned"]
        calibrated = np.einsum("sij,nsj->nsi", matrices, aligned) + offsets
        stored_calibrated = np.einsum("sij,nsj->nsi", stored_matrices, aligned) + stored_offsets
        pre = delta_o(aligned) * 100  # 1 G = 100 microtesla
        post = delta_o(calibrated) * 100
        all_pre.append(pre)
        all_post.append(post)
        sample_rows.extend({
            "channel": cfg["channel"], "voltage_V": cfg["voltage_V"],
            "sample_index": k + 1, "pre_uT": float(pre[k]), "post_uT": float(post[k]),
        } for k in range(len(pre)))
        rows.append({
            "channel": cfg["channel"], "voltage_V": cfg["voltage_V"],
            "samples": len(aligned),
            "mean_reference_mT": cfg["mean_reference_G"] * 0.1,
            "pre_mean_uT": float(pre.mean()), "pre_std_uT": float(pre.std()),
            "post_refit_mean_uT": float(post.mean()), "post_refit_std_uT": float(post.std()),
            "post_stored_mean_uT": float((delta_o(stored_calibrated) * 100).mean()),
        })
    summary = {
        "numpy_version": np.__version__, "samples": sum(r["samples"] for r in rows),
        "sensors": 12, "files": len(rows), "weights": weights.tolist(),
        "augmented_design_ranks": ranks, "per_sensor_component_rmse_G": rmse,
        "stored_parameter_max_abs_difference": {
            "D": float(np.max(np.abs(matrices - stored_matrices))),
            "e_G": float(np.max(np.abs(offsets - stored_offsets))),
        },
        "equal_weight_condition_statistics_uT": {
            "pre_mean": float(np.mean([r["pre_mean_uT"] for r in rows])),
            "pre_std_ddof0": float(np.std([r["pre_mean_uT"] for r in rows])),
            "pre_std_ddof1": float(np.std([r["pre_mean_uT"] for r in rows], ddof=1)),
            "post_mean": float(np.mean([r["post_refit_mean_uT"] for r in rows])),
            "post_std_ddof0": float(np.std([r["post_refit_mean_uT"] for r in rows])),
            "post_std_ddof1": float(np.std([r["post_refit_mean_uT"] for r in rows], ddof=1)),
        },
        "pooled_sample_statistics_uT": {
            "pre_mean": float(np.concatenate(all_pre).mean()),
            "pre_std_ddof0": float(np.concatenate(all_pre).std()),
            "post_mean": float(np.concatenate(all_post).mean()),
            "post_std_ddof0": float(np.concatenate(all_post).std()),
        },
        "evaluation": "Residual consistency on the same 15 files used to fit the model.",
    }
    output.mkdir(parents=True, exist_ok=True)
    parameters = {str(i + 1): {"D_i": matrices[i].tolist(), "e_i": offsets[i].tolist()}
                  for i in range(12)}
    (output / "affine_model_params_refit.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "per_condition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
