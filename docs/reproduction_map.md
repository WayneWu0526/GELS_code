# Results and their files

| Result | Entry point | Frozen inputs | Reference output |
|---|---|---|---|
| Minimal algorithm example | `core/run_simulation.py` | `core/configs/example.yaml` | Expected errors in `core/README.md` |
| Affine D_i and e_i | `calibration/reproduce_calibration.py` | `calibration/data/`, `calibration/config/` | `calibration/reference_results/affine_model_params_refit.json` |
| Calibration consistency curves | `calibration/plot_residuals.py` | `per_sample.csv`, `summary.json` from fit | `calibration/reference_results/residual_consistency.pdf` |
| Continuous source-shape sweep | `source_geometry/run_gshape_sweep.py` | `config/gshape_main.yaml`, frozen sensor geometry | `source_geometry/reference_results/gshape_main/` |
| Supplementary Fig. 1 and Table I numbers | `source_geometry/run_trajectory_localization.py` | Five saved source configurations and `config/trajectory_localization.yaml` | `source_geometry/reference_results/trajectory_localization/` |
| Supplementary Fig. 1 composition | `source_geometry/plot_trajectory_localization_2x5.py` | Numerical outputs plus five PDF-extracted RViz views | Existing figure PDF and extracted views included. Original screenshot files are not included |
| Supplementary Fig. 2 | `distance/run_distance_sweep.py`, `distance/export_paper_tables.py` | `simulation_parameters.json`, `config.json` | `distance/reference_results/`, `distance/paper_figure/` |
| Supplementary Fig. 3 | `tdm_motion/run_circle.py`, `tdm_motion/render_figures.py` | `tdm_motion/config.json`, `tdm_motion/reference/` | `tdm_motion/reference_results/` |
| Literature comparison Table II | `supplementary/Supplementary.tex` | Cited literature in `references.bbl` | `Supplementary.pdf` |

Each path in this table is relative to the repository root. Calibration measurements, fixed parameters derived from prior measurements, synthetic results, and the literature comparison have distinct origins. The package reproduces simulations and calibration, not the manuscript's localization experiments.
