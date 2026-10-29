# Minimal GELS example

This is a small synthetic example of the localization algorithm. It is not a replay of the paper's measured experiments.

From the repository root:

```bash
python3 core/run_simulation.py --config core/configs/example.yaml
```

The default noiseless example uses 12 sensors and three point dipoles. Its position error is approximately **0.0463 mm**, and its orientation error is approximately **0.0534 degrees**. The finite array approximates a local field gradient, so noiseless sampling does not imply zero pose error.

`maps_estimator.py` recovers the local field and symmetric traceless gradient, estimates source-relative vectors, and uses `kabsch_solver.py` to recover the array pose. The historical function name `MaPS_Estimator` is retained. `simulation.py` and `mag_dipole_model.py` generate synthetic observations.

`configs/example.yaml` specifies sensor offsets, target pose, source positions and moments, model order, and optional noise. Positions use metres, fields use tesla, and source moments use A m². Input Euler angles use degrees in the order `[yaw, pitch, roll]`, with `R = Rz @ Ry @ Rx`. The rotation maps the sensor frame into the world frame.

The estimator takes `D_cal` with shape `(3, N)`, one source-position dictionary per source, and one `(3, N)` field matrix per source. In this interface, `D_cal` means sensor offsets. It is distinct from the per-sensor affine calibration matrices `D_i` in the calibration module. Returned rotation and position describe the array pose in the world frame.

The study modules preserve their original estimator snapshots. Their provenance and numerical checks are recorded in `docs/`.
