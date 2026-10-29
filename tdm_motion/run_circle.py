"""Phase-matched, noiseless circular translation with twelve sensors only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'reference'))
from core import (load_config, load_inputs, sensor_offsets, schedule,
                  sample_fields, estimate, errors, kabsch_solver)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=plain, allow_nan=False)+'\n')


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def circle_pose(center, radius, R, phase, speed_m_s, reference_s):
    """Constant scalar speed, changing velocity direction; no body rotation."""
    def pose(t):
        angle = phase + speed_m_s/radius*(t-reference_s)
        return center+radius*np.array([np.cos(angle), np.sin(angle), 0.]), R.copy()
    return pose


def geometric_estimate(sources, pose_at, source_times):
    rho = np.column_stack([pose_at(t)[1].T@(pose_at(t)[0]-source['p_Ci'])
                           for source, t in zip(sources, source_times)])
    c = np.column_stack([s['p_Ci'] for s in sources])
    Rh = kabsch_solver(c-c.mean(axis=1, keepdims=True), -(rho-rho.mean(axis=1, keepdims=True)))
    return Rh, c.mean(axis=1)+Rh@rho.mean(axis=1), rho


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'results')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE/'config.json').read_text())
    ref_cfg = load_config()
    D = sensor_offsets(ref_cfg, cfg['configuration'])
    assert D.shape == (3, 12) and cfg['sensor_count'] == 12
    sources, targets = load_inputs(ref_cfg)
    center = np.mean([p['p'] for p in targets], axis=0)
    R = next(p['R'] for p in targets if p['index'] == 0)
    radius = cfg['trajectory']['radius_m']
    phases = np.arange(cfg['trajectory']['phase_count'])*2*np.pi/cfg['trajectory']['phase_count']
    timings = {mode: schedule(ref_cfg, cfg['configuration'], mode) for mode in cfg['modes']}
    geometry = dict(center_m=center, radius_m=radius, fixed_R=R,
                    source_positions_m=[s['p_Ci'] for s in sources],
                    source_moments_A_m2=[s['m_Ci'] for s in sources], sensor_offsets_m=D.T)
    save_json(out/'geometry.json', geometry)
    save_json(out/'config_snapshot.json', cfg)
    save_json(out/'baseline_config_snapshot.json', ref_cfg)
    save_json(out/'timing_profiles.json', timings)
    path_rows = []
    for i, phase in enumerate(phases):
        p, _ = circle_pose(center, radius, R, phase, 0., 0.)(0.)
        path_rows.append(dict(phase_index=i, phase_deg=np.degrees(phase), x_m=p[0], y_m=p[1], z_m=p[2],
                              x_mm=p[0]*1000, y_mm=p[1]*1000, z_mm=p[2]*1000))
    save_csv(out/'reference_trajectory.csv', path_rows)
    save_csv(out/'sources.csv', [dict(source=k+1, x_mm=s['p_Ci'][0]*1000,
                                     y_mm=s['p_Ci'][1]*1000, z_mm=s['p_Ci'][2]*1000)
                               for k, s in enumerate(sources)])
    rows, p_est, R_est, rho_est = [], [], [], []
    fields, field_rows = [], []
    static_sync = {}
    for i, phase in enumerate(phases):
        pose = circle_pose(center, radius, R, phase, 0., 0.)
        F, _ = sample_fields(D, sources, pose, timings['CV'], 'point_dipole_full_array', 'synchronous')
        Rh, ph, details = estimate(D, sources, F)
        static_sync[i] = (np.array(F), Rh, ph, details)
    max_zero_field_delta = 0.
    max_repeat_position_delta = 0.
    max_orthogonality_error = 0.
    max_radius_error = 0.
    started = time.monotonic()
    for speed in cfg['trajectory']['speed_mm_s']:
        for mode, timing in timings.items():
            slot_timing = dict(timing)
            slot_timing['source_sample_times_s'] = np.repeat(timing['source_effective_times_s'][:,None], 12, axis=1)
            tref = timing['acquisition_end_s']
            for phase_index, phase in enumerate(phases):
                pose = circle_pose(center, radius, R, phase, speed/1000, tref)
                p, Rt = pose(tref)
                p_end, R_end = pose(timing['cycle_end_s'])
                sync_F, sync_R, sync_p, sync_details = static_sync[phase_index]
                for acquisition in ['synchronous', 'source_slot_only', 'sequential_sensor', 'ideal_rho']:
                    if acquisition == 'synchronous':
                        F, Rh, ph, details = sync_F, sync_R, sync_p, sync_details
                        rho = details['rho_hats']; cond = max(np.linalg.cond(x) for x in details['X_hat_locals'])
                    elif acquisition == 'ideal_rho':
                        Rh, ph, rho = geometric_estimate(sources, pose, timing['source_effective_times_s'])
                        F, cond = None, None
                    else:
                        use_timing = timing if acquisition == 'sequential_sensor' else slot_timing
                        F, _ = sample_fields(D, sources, pose, use_timing, 'point_dipole_full_array', 'sequential')
                        Rh, ph, details = estimate(D, sources, F)
                        rho = details['rho_hats']; cond = max(np.linalg.cond(x) for x in details['X_hat_locals'])
                        if speed == 0:
                            max_zero_field_delta = max(max_zero_field_delta, float(np.max(np.abs(np.array(F)-sync_F))))
                        # Replay fixed phases as an independent determinism check.
                        if acquisition == 'sequential_sensor' and phase_index in (0, 47, 119):
                            F2, _ = sample_fields(D, sources, pose, use_timing, 'point_dipole_full_array', 'sequential')
                            R2, p2, _ = estimate(D, sources, F2)
                            assert np.array_equal(F, F2) and np.array_equal(Rh, R2)
                            max_repeat_position_delta = max(max_repeat_position_delta, float(np.linalg.norm(ph-p2)))
                    ep, er = errors(ph, Rh, p, Rt)
                    ep_end, er_end = errors(ph, Rh, p_end, R_end)
                    dp, dr = errors(ph, Rh, sync_p, sync_R)
                    assert np.all(np.isfinite(ph)) and np.all(np.isfinite(Rh)) and np.all(np.isfinite(rho))
                    max_orthogonality_error = max(max_orthogonality_error, float(np.linalg.norm(Rh.T@Rh-np.eye(3))))
                    row_index = len(rows)
                    rows.append(dict(case=row_index, speed_mm_s=speed, mode=mode, acquisition=acquisition,
                                     phase_index=phase_index, phase_deg=float(np.degrees(phase)),
                                     ep_mm=ep, er_deg=er, ep_cycle_end_mm=ep_end, er_cycle_end_deg=er_end,
                                     paired_position_shift_mm=dp, paired_rotation_shift_deg=dr,
                                     max_gradient_condition=cond,
                                     reference_time_s=tref, cycle_end_s=timing['cycle_end_s']))
                    p_est.append(ph); R_est.append(Rh); rho_est.append(rho)
                    if F is not None:
                        fields.append(F); field_rows.append(row_index)
                for t in timing['source_sample_times_s'].ravel():
                    pt, _ = pose(t)
                    max_radius_error = max(max_radius_error, abs(float(np.linalg.norm(pt-center))-radius))
            print(f'{speed:>3} mm/s {mode}: {len(phases)} phases complete', flush=True)
    save_csv(out/'cases.csv', rows)
    np.savez_compressed(out/'estimates.npz', p_est_m=p_est, R_est=R_est, rho_est_m=rho_est)
    np.savez_compressed(out/'measurements.npz', case_indices=field_rows, fields_T=fields)
    metrics = ['ep_mm', 'er_deg', 'ep_cycle_end_mm', 'paired_position_shift_mm', 'paired_rotation_shift_deg']
    summary = []
    for speed in cfg['trajectory']['speed_mm_s']:
        for mode in cfg['modes']:
            for acquisition in ['synchronous','source_slot_only','sequential_sensor','ideal_rho']:
                group = [r for r in rows if r['speed_mm_s'] == speed and r['mode'] == mode and r['acquisition'] == acquisition]
                item = dict(speed_mm_s=speed, mode=mode, acquisition=acquisition, phase_count=len(group))
                for metric in metrics:
                    values = np.array([r[metric] for r in group])
                    item.update({metric+'_mean':float(values.mean()), metric+'_median':float(np.median(values)),
                                 metric+'_p95':float(np.quantile(values,.95)), metric+'_max':float(values.max())})
                summary.append(item)
    save_csv(out/'summary.csv', summary)
    for acquisition in ['synchronous','source_slot_only','sequential_sensor','ideal_rho']:
        for mode in cfg['modes']:
            save_csv(out/f'{mode}_{acquisition}.csv', [r for r in summary if r['mode']==mode and r['acquisition']==acquisition])
    zero = [r for r in rows if r['speed_mm_s']==0]
    for mode in cfg['modes']:
        sync = [r for r in zero if r['mode']==mode and r['acquisition']=='synchronous']
        seq = [r for r in zero if r['mode']==mode and r['acquisition']=='sequential_sensor']
        assert all(a['ep_mm']==b['ep_mm'] and a['er_deg']==b['er_deg'] for a,b in zip(sync,seq))
    assert max_zero_field_delta == 0. and max_repeat_position_delta == 0.
    assert max_orthogonality_error < 1e-12 and max_radius_error < 1e-12
    # Independent speed check using finite differences along the trajectory.
    pose = circle_pose(center, radius, R, .73, .1, timings['CV']['acquisition_end_s'])
    h = 1e-6
    measured_speed = np.linalg.norm((pose(h)[0]-pose(-h)[0])/(2*h))
    assert abs(measured_speed-.1) < 1e-8
    ideal_zero = [r for r in zero if r['acquisition']=='ideal_rho']
    assert max(r['ep_mm'] for r in ideal_zero) < 1e-8
    # Exact rho must remain independent of source moment amplitude by construction.
    c = np.array([s['p_Ci'] for s in sources])
    positions = np.array([[r['x_m'],r['y_m'],r['z_m']] for r in path_rows])
    distance = np.linalg.norm(positions[:,None,:]-c[None,:,:],axis=2)
    checks = dict(status='passed', case_count=len(rows), phase_count=len(phases), speed_count=len(cfg['trajectory']['speed_mm_s']),
                  nonfinite_results=0, discarded_cases=0,
                  max_zero_speed_field_difference_T=max_zero_field_delta,
                  max_repeat_position_difference_m=max_repeat_position_delta,
                  checked_repeat_phases=[0,47,119],
                  max_rotation_orthogonality_error=max_orthogonality_error, max_path_radius_error_m=max_radius_error,
                  finite_difference_speed_m_s=measured_speed, ideal_rho_static_max_ep_mm=max(r['ep_mm'] for r in ideal_zero),
                  source_distance_range_mm=[float(distance.min()*1000),float(distance.max()*1000)],
                  phase_statistics_note='deterministic phase distribution, not repeated independent noisy trials',
                  elapsed_s=time.monotonic()-started)
    save_json(out/'validation.json',checks)
    inputs = [BASE/'config.json',Path(__file__).resolve()]+[p for p in (BASE/'reference').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    save_json(out/'manifest.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
             input_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
             output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))
    print('All cases and validation saved to',out,flush=True)


if __name__=='__main__':
    main()
