# Motion during sequential acquisition

This noiseless point-dipole study uses 12 sensors, fixed source geometry, a circular translation of radius 40 mm, and a fixed array orientation. It evaluates eight speeds `(0, 1, 2, 5, 10, 20, 50, 100)` mm/s at 180 matched reference phases, for CV and CC acquisition and four sampling models: 11,520 cases in total.

The nominal CV schedule uses four 72 ms slots (288 ms per cycle). CC uses three 32 ms slots (96 ms per cycle). Each sensor is sampled at the midpoint of its successive 1 ms window. These timestamps are stated model assumptions. They are not a replay of firmware timing. Truth is evaluated at the end of the last active-source acquisition window.

Run from the repository root:

```bash
python3 tdm_motion/run_circle.py --output tdm_motion/results
```

The main model is `sequential_sensor`. Controls make all sensors simultaneous within each source slot (`source_slot_only`), sample every source at the same time (`synchronous`), or replace field inversion by exact relative vectors (`ideal_rho`). The synchronous model retains the finite-array spatial approximation.

The paper plot shows 0–10 mm/s. At 10 mm/s, the sequential model gives mean position/orientation errors of approximately 1.8266 mm / 1.0041 degrees for CV and 1.5400 mm / 1.0172 degrees for CC. The 5th–95th percentile bars describe variation across deterministic trajectory phases. They are not confidence intervals or noise statistics.

`reference/` freezes the estimator, source geometry, sensor layout, and timing inputs. `reference_results/` contains all cases, summaries, reconstructed poses, sampled fields, and published figure data.

With LaTeX, PGFPlots, and Poppler (`pdfinfo` and `pdftoppm`) available:

```bash
python3 tdm_motion/render_figures.py --results tdm_motion/results
```

This regenerates `figures/dynamic_errors.pdf`, which is the three-panel supplementary figure, and a separate sampling-control diagnostic. The renderer derives its main trajectory radius from the numerical geometry output.
