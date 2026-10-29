"""
simulation.py: Generate synthetic magnetic field measurements from config.

Coordinate convention:
  - All positions in meters, fields in Tesla.
  - Sensor array pose given as (p_target, R_target) in the global frame.
  - Sensor j global position: p_sensor_j = p_target + R_target @ d_j
  - Source i global field at sensor j: mag_dipole_model(p_sensor_j, m_i, p_Ci)
  - Local (sensor-frame) measurement: B_local_j = R_target.T @ B_global_j
"""
import numpy as np
from mag_dipole_model import mag_dipole_model, eul2rotm


def build_sources(sources_cfg: list) -> list:
    """
    Parse source config entries into {'p_Ci': (3,), 'm_Ci': (3,)} dicts.

    Each entry must have:
        position:          [x, y, z] in meters
        moment_direction:  [dx, dy, dz]  (will be normalised)
        moment_magnitude:  scalar in A·m²
    """
    sources = []
    for i, src in enumerate(sources_cfg):
        p_Ci = np.array(src['position'], dtype=float)
        direction = np.array(src['moment_direction'], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(f"Source {i}: moment_direction has zero norm.")
        m_Ci = (direction / norm) * float(src['moment_magnitude'])
        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci})
    return sources


def simulate_measurements(d_list: np.ndarray, p_target: np.ndarray,
                           R_target: np.ndarray, sources: list,
                           noise_std: float = 0.0, rng: np.random.Generator = None,
                           dipole_order: int = 1) -> list:
    """
    Generate B_meas_cell for each source.

    Args:
        d_list:      (N, 3) sensor offsets in sensor frame (meters)
        p_target:    (3,)  sensor array center position (global frame, meters)
        R_target:    (3,3) sensor array orientation (global frame)
        sources:     list of M dicts with 'p_Ci' and 'm_Ci'
        noise_std:   Gaussian noise std-dev in Tesla (0 = noiseless)
        rng:         numpy random Generator for reproducible noise
        dipole_order: passed to mag_dipole_model (1, 3, or 5)

    Returns:
        B_meas_cell: list of M (3, N) arrays in Tesla (sensor frame)
    """
    d_list = np.asarray(d_list, dtype=float)   # (N, 3)
    N = d_list.shape[0]
    M = len(sources)
    B_meas_cell = []

    for src in sources:
        B_meas = np.zeros((3, N))
        for j in range(N):
            d_j = d_list[j]
            p_sensor_j = p_target + R_target @ d_j             # global position
            b_global, _ = mag_dipole_model(
                p_sensor_j, src['m_Ci'], src['p_Ci'], order=dipole_order
            )
            b_local = R_target.T @ b_global                    # sensor frame
            B_meas[:, j] = b_local

        if noise_std > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            B_meas += rng.standard_normal((3, N)) * noise_std

        B_meas_cell.append(B_meas)

    return B_meas_cell


def run_from_config(cfg: dict):
    """
    Run a full simulation from a parsed config dict.

    Returns a dict with:
        D_cal:        (3, N)
        sources:      list of M dicts
        B_meas_cell:  list of M (3, N) arrays
        p_target:     (3,)
        R_target:     (3, 3)
        p_gt:         ground-truth position (same as p_target)
        R_gt:         ground-truth rotation (same as R_target)
    """
    # --- Sensor array ---
    arr_cfg = cfg['sensor_array']
    d_list_full = np.array(arr_cfg['d_list'], dtype=float)   # (N_full, 3)

    sensor_ids = arr_cfg.get('sensor_ids', None)
    if sensor_ids is not None:
        indices = [sid - 1 for sid in sensor_ids]
        d_list = d_list_full[indices]
    else:
        d_list = d_list_full

    N = d_list.shape[0]
    D_cal = d_list.T.copy()   # (3, N)

    # --- Target pose ---
    pose_cfg = cfg['target_pose']
    p_target = np.array(pose_cfg['position'], dtype=float)
    euler_deg = np.array(pose_cfg['euler_zyx_deg'], dtype=float)
    R_target = eul2rotm(np.deg2rad(euler_deg))

    # --- Sources ---
    sources = build_sources(cfg['sources'])

    # --- Simulation params ---
    sim_cfg = cfg.get('simulation', {})
    noise_std = float(sim_cfg.get('noise_std_tesla', 0.0))
    seed = sim_cfg.get('random_seed', None)
    dipole_order = int(sim_cfg.get('dipole_order', 1))

    rng = np.random.default_rng(seed) if noise_std > 0.0 else None

    B_meas_cell = simulate_measurements(
        d_list, p_target, R_target, sources,
        noise_std=noise_std, rng=rng, dipole_order=dipole_order
    )

    return {
        'D_cal': D_cal,
        'sources': sources,
        'B_meas_cell': B_meas_cell,
        'p_target': p_target,
        'R_target': R_target,
        'p_gt': p_target,
        'R_gt': R_target,
        'd_list': d_list,
        'N': N,
        'M': len(sources),
    }
