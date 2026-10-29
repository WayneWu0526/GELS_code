# Source geometry

This module contains two separate point-dipole simulation studies. All position values use metres internally. The field noise is an assumed independent Gaussian standard deviation of 0.5 μT per axis and source. Each source has a moment magnitude of 250 A m².

## Continuous shape sweep

`source_geometry.py` constructs three centered source positions with shape ratio `g = sigma_2 / sigma_1`. The construction fixes the source centroid and the Frobenius norm of the centered position matrix at 0.33 m. Individual source-to-target distances and local field strengths still change with the geometry.

The sweep uses 25 shape values, the same 100 deterministic Latin-hypercube target poses in an 80 × 80 × 80 mm volume centered at `(0, 0, -0.30)` m, and 30 noise realizations per pose: 75,000 localization cases in total. Positions and noise are shared across shape values. Orientations are random with a fixed seed.

Run from the repository root:

```bash
python3 source_geometry/run_gshape_sweep.py --config source_geometry/config/gshape_main.yaml --output source_geometry/results/gshape_main
```

The archived endpoint means are 12.50 mm / 2.74 degrees at `g=1` and 307.73 mm / 70.06 degrees at `g=0.01`.

For each target pose, errors are first averaged across its 30 realizations. The reported mean and P10–P90 then summarize those per-pose means. These quantiles describe variation across poses.

## Five recorded source configurations

`inputs/moveit_frustum_search.json` preserves five geometries from a prior collision-checked MoveIt planning run. It records `MOVEIT_FAKE_ONLY`, `execute_called=false`, and no physical trajectory execution. This module reads the saved source positions and axes. It neither runs MoveIt nor implements the proposed projected-ascent update in the main manuscript.

The evaluation uses 200 target poses in a 240 × 240 × 80 mm volume and 30 noise realizations per pose, giving 30,000 cases across five states. The sampled z range is approximately 0.750447–0.830447 m. The figure's wider z axis is a display range.

```bash
python3 source_geometry/run_trajectory_localization.py --config source_geometry/config/trajectory_localization.yaml --output source_geometry/results/trajectory_localization
```

| g | Mean position error (mm) | Mean orientation error (degrees) |
|---:|---:|---:|
| 1 | 11.76 | 2.93 |
| 0.5 | 13.96 | 3.41 |
| 0.2 | 26.85 | 5.78 |
| 0.05 | 109.11 | 22.43 |
| 0.01 | 290.56 | 68.11 |

`reference_results/` contains all trial errors, per-pose means, target poses, source configurations, summary tables, and prior reproduction records. Configurations and seeds are under `config/`. `vendor/` preserves the numerical estimator, dipole model, and array configuration used by these studies.

## Figures

The current supplementary figure is supplied as `../supplementary/figures/supp_source_geometry_sequence.pdf`. Numerical results are fully included.

With Matplotlib installed, the continuous-sweep plot can be regenerated from its summary:

```bash
python3 source_geometry/plot_gshape_results.py --summary source_geometry/reference_results/gshape_main/summary_by_gshape.csv --output-stem source_geometry/results/geometry_errors
```

`plot_trajectory_localization_2x5.py` retains its historical filename and generates the final three-row layout. The five images in `inputs/rviz_from_pdf/` were extracted from the image streams embedded in the existing figure PDF, in state order. They preserve the figure-resolution RViz views. The original screenshot files are not included. Their source PDF hash and image hashes are recorded in `provenance.json` in that folder. Regeneration may differ slightly in raster resampling and font metrics.

```bash
python3 source_geometry/plot_trajectory_localization_2x5.py --results source_geometry/reference_results/trajectory_localization --rviz-images source_geometry/inputs/rviz_from_pdf/state_0.png source_geometry/inputs/rviz_from_pdf/state_1.png source_geometry/inputs/rviz_from_pdf/state_2.png source_geometry/inputs/rviz_from_pdf/state_3.png source_geometry/inputs/rviz_from_pdf/state_4.png --output-stem source_geometry/results/representative_geometries
```

The archived figure remains the version used by the supplementary document. No robot model or planning environment is required for the numerical reproduction or figure composition.
