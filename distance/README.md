# Source distance and gradient resolvability

This module reproduces the point-dipole distance simulations from fixed inputs. It includes both the outward ray and the control simulations at ten reference geometries. All generated fields and estimated poses are synthetic.

## Parameters and scope

`simulation_parameters.json` contains sensor coordinates, source positions and axes, reference geometries, and two fixed source-moment settings. The geometric parameters were derived from tracked poses. The nominal setting gives a mean field of 20 G (2 mT) at the original reference geometries. The field-fit setting contains three signed scalar moments previously fitted to measured field vectors. Their origin is recorded in the parameter file. This distribution reproduces the simulations with those constants. It does not reproduce the original magnetic-moment fit or the Fig. 6 localization measurements.

The Config. 2 paper curves use eight vertex sensors, 41 distances from approximately 205.9 to 800 mm, and 200 realizations per noisy condition. Raw-slot noise has a standard deviation of 0.3 μT per sensor axis. Each source-separated residual has vector RMS 6 μT and zero spatial mean. Noise and residual values are sensitivity assumptions. The random seed, draw order, estimator, and numerical settings are unchanged.

## Reproduce

Run from the repository root, using a fresh output directory:

```bash
python3 distance/run_distance_sweep.py --output distance/results
python3 distance/export_paper_tables.py --results distance/results --output distance/results/paper_figure
```

The full run produces 81,804 synthetic trials and 612 summary rows, including Config. 2 and Config. 3, both moment settings, and ideal/noise/residual controls. The legacy table label `domain=recorded` means synthetic fields evaluated at the fixed reference geometries. It does not denote measured field samples or experimental localization estimates. `--smoke` reduces the counts and is only a quick execution check.

`reference_results/` preserves the full simulation tables and ray geometry. `paper_figure/` includes the 41-point plotting tables. With PGFPlots installed:

```bash
cd distance/results/paper_figure
pdflatex -interaction=nonstopmode -halt-on-error supp_distance_extension.tex
```

The solid gradient-quality curves use `-20 log10(RMS(relative gradient error))` across the trials. Dashed curves use the random-noise gradient SNR. Position bands are P10–P90 across realizations.

With the plotting dependencies installed:

```bash
python3 distance/plot_results.py distance/results
```

This produces simulation diagnostics only. No raw localization acquisitions, historical estimates, replay tables, or measured/simulated overlays are included.
