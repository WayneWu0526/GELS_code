"""Noiseless GELS forward model and nominal TDM schedule. SI units throughout."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "vendor"))
from mag_dipole_model import mag_dipole_model
from maps_estimator import MaPS_Estimator, compute_rho_hat_eigen
from kabsch_solver import kabsch_solver


def load_config(path=None):
    return json.loads(Path(path or BASE / "config.json").read_text())


def rotation_from_quaternion(q):
    q = np.asarray(q, dtype=float)
    if np.linalg.norm(q) == 0:
        raise ValueError("Zero quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def load_inputs(cfg):
    with (BASE / cfg["sources_csv"]).open() as f:
        rows = list(csv.DictReader(f))
    sources = []
    for row, amplitude in zip(rows, cfg["source_signed_moments_A_m2"]):
        axis = np.array([float(row[k]) for k in ("axis_x", "axis_y", "axis_z")])
        sources.append({
            "p_Ci": np.array([float(row[k]) for k in ("x_m", "y_m", "z_m")]),
            "m_Ci": amplitude * axis / np.linalg.norm(axis),
        })
    if len(sources) != 3:
        raise ValueError("This baseline requires exactly three sources")
    with (BASE / cfg["targets_csv"]).open() as f:
        rows = list(csv.DictReader(f))
    targets = [dict(
        index=int(r["target_index"]),
        p=np.array([float(r[k]) for k in ("x_m", "y_m", "z_m")]),
        R=rotation_from_quaternion([float(r[k]) for k in ("qx", "qy", "qz", "qw")]),
    ) for r in rows]
    return sources, targets


def sensor_offsets(cfg, configuration):
    ids = cfg["configurations"][configuration]
    return np.asarray(cfg["sensor_offsets_m"])[np.array(ids)-1].T


def schedule(cfg, configuration, mode):
    """Nominal point-sample times, not measured firmware timestamps.

    Sensor acquisition windows are consecutive and their midpoints define the
    samples. A moving-pose callback can subsequently replace the static pose.
    """
    timing = cfg["timing"]
    ids = cfg["configurations"][configuration]
    n = len(ids)
    step = 1.0 / timing["sensor_rate_Hz"]
    if mode == "CV":
        wait = timing["CV_wait_s"]
        slot = timing["CV_slot_s"]
        count = timing["CV_slots"]
    elif mode == "CC":
        wait = timing["CC_wait_s"]
        slot = wait + n*step
        count = timing["CC_slots"]
    else:
        raise ValueError(mode)
    if wait + n*step > slot + 1e-14:
        raise ValueError("Sensor acquisition does not fit in slot")
    times = np.arange(count)[:, None]*slot + wait + (np.arange(n)+0.5)*step
    return dict(
        configuration=configuration, mode=mode, sensor_ids=ids,
        slot_s=slot, wait_s=wait, sample_step_s=step,
        cycle_s=count*slot, update_Hz=1/(count*slot),
        source_sample_times_s=times[:3],
        background_sample_times_s=times[3] if count == 4 else None,
        source_effective_times_s=times[:3].mean(axis=1),
        last_source_sample_s=float(times[2, -1]),
        acquisition_end_s=2*slot+wait+n*step,
        cycle_end_s=count*slot,
    )


def stationary_pose(p, R):
    p, R = np.array(p, copy=True), np.array(R, copy=True)
    return lambda t: (p.copy(), R.copy())


def sample_fields(D, sources, pose_at, timing, model, acquisition="sequential"):
    """Generate three source-separated field matrices and CV background.

    pose_at(t) returns array-center p and sensor-to-world R. It may describe an
    arbitrary trajectory, including rotation. This stage runs stationary poses
    only. No trajectory is reconstructed from sparse archived records.
    """
    if acquisition not in ("sequential", "synchronous"):
        raise ValueError(acquisition)
    if model not in ("point_dipole_full_array", "matched_local_linear"):
        raise ValueError(model)
    ref = timing["acquisition_end_s"]
    fields = []
    for k, source in enumerate(sources):
        B = np.empty_like(D)
        for j, d in enumerate(D.T):
            t = timing["source_sample_times_s"][k, j] if acquisition == "sequential" else ref
            p, R = pose_at(float(t))
            if model == "point_dipole_full_array":
                b, _ = mag_dipole_model(p+R@d, source["m_Ci"], source["p_Ci"], order=1)
                B[:, j] = R.T@b
            else:
                b, A = mag_dipole_model(p, source["m_Ci"], source["p_Ci"], order=1)
                B[:, j] = R.T@b + (R.T@A@R)@d
        fields.append(B)
    background = np.zeros_like(D) if timing["mode"] == "CV" else None
    separated = [B-background for B in fields] if background is not None else fields
    return separated, background


def estimate(D, sources, fields):
    return MaPS_Estimator(D, sources, fields)


def recover_from_analytic_quantities(sources, p, R):
    """Independent center-field/gradient check, bypassing finite array fitting."""
    rho, euler_residual = [], []
    for source in sources:
        b, A = mag_dipole_model(p, source["m_Ci"], source["p_Ci"])
        b, X = R.T@b, R.T@A@R
        rho.append(compute_rho_hat_eigen(X, b))
        euler_residual.append(np.linalg.norm(X@(R.T@(p-source["p_Ci"]))+3*b))
    rho = np.column_stack(rho)
    centers = np.column_stack([s["p_Ci"] for s in sources])
    Rh = kabsch_solver(centers-centers.mean(axis=1, keepdims=True),
                       -(rho-rho.mean(axis=1, keepdims=True)))
    ph = centers.mean(axis=1)+Rh@rho.mean(axis=1)
    return Rh, ph, max(euler_residual)


def errors(ph, Rh, p, R):
    Q = Rh@R.T
    # atan2 avoids acos roundoff floors near zero; same SO(3) geodesic angle.
    sine = np.linalg.norm([Q[2, 1]-Q[1, 2], Q[0, 2]-Q[2, 0], Q[1, 0]-Q[0, 1]])/2
    angle = np.arctan2(sine, (np.trace(Q)-1)/2)
    return float(np.linalg.norm(ph-p)*1000), float(np.degrees(angle))
