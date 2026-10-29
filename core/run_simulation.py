#!/usr/bin/env python3
"""
run_simulation.py: CLI entry point for the MaPS standalone simulation.

Usage:
    python run_simulation.py --config configs/example.yaml
    python run_simulation.py --config configs/example.yaml --output result.json
"""
import argparse
import json
import sys

import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required.  Install with:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from simulation import run_from_config
from maps_estimator import MaPS_Estimator
from mag_dipole_model import rotm2eul_zyx


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    R_diff = R_est.T @ R_gt
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def main():
    parser = argparse.ArgumentParser(
        description="MaPS standalone simulation — reproduces 6-DoF pose estimation."
    )
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', default=None, help='Optional path to save results as JSON')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Run simulation (generates synthetic B_meas_cell from ground-truth pose)
    sim = run_from_config(cfg)
    D_cal = sim['D_cal']
    sources = sim['sources']
    B_meas_cell = sim['B_meas_cell']
    p_gt = sim['p_gt']
    R_gt = sim['R_gt']

    print(f"Sensor count N = {sim['N']},  Source count M = {sim['M']}")
    print(f"Ground-truth position  [m]:  {p_gt}")
    print(f"Ground-truth ZYX euler [deg]: {np.degrees(rotm2eul_zyx(R_gt))}")
    print()

    # Run MaPS estimator
    R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)

    # Compute errors
    pos_err_m = float(np.linalg.norm(p_est - p_gt))
    pos_err_mm = pos_err_m * 1e3
    rot_err_deg = rotation_error_deg(R_est, R_gt)

    print("=== MaPS Estimation Results ===")
    print(f"Estimated position     [m]:   {p_est}")
    print(f"Estimated ZYX euler    [deg]: {np.degrees(rotm2eul_zyx(R_est))}")
    print()
    print(f"Position error:  {pos_err_mm:.4f} mm")
    print(f"Rotation error:  {rot_err_deg:.4f} deg")
    print()

    print("--- Intermediate quantities ---")
    rho_hats = details['rho_hats']
    b_hat_locals = details['b_hat_locals']
    for i in range(sim['M']):
        print(f"  Source {i}: rho_hat = {rho_hats[:, i] * 1e3}  mm  |  "
              f"|b_hat| = {np.linalg.norm(b_hat_locals[:, i]) * 1e6:.4f} uT")

    # Optionally save JSON output
    if args.output:
        result = {
            'config': args.config,
            'N': sim['N'],
            'M': sim['M'],
            'ground_truth': {
                'position': p_gt.tolist(),
                'euler_zyx_deg': np.degrees(rotm2eul_zyx(R_gt)).tolist(),
                'R': R_gt.tolist(),
            },
            'estimate': {
                'position': p_est.tolist(),
                'euler_zyx_deg': np.degrees(rotm2eul_zyx(R_est)).tolist(),
                'R': R_est.tolist(),
            },
            'errors': {
                'position_mm': pos_err_mm,
                'rotation_deg': rot_err_deg,
            },
            'rho_hats_mm': (rho_hats * 1e3).tolist(),
            'b_hat_locals_uT': (b_hat_locals * 1e6).tolist(),
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
