# GELS: gradiometer-based electromagnetic localization

Code, recorded calibration data, simulation inputs, and supplementary material accompanying:

**Closed-Form 6-DoF Pose Estimation of Medical Tools via Gradiometer-Based Electromagnetic Localization System**  
Zhiwei Wu, Jiahao Luo, Yubo Pu, Siyi Wei, Yuankai Chen, and Jinhui Zhang

Read the [three-page supplementary material](Supplementary.pdf). The repository provides an introductory estimator example and four independent reproduction modules. Each module includes its own frozen configuration and dependencies so that a change in one study does not silently change another.

## Contents

| Directory | Contents | Paper connection |
|---|---|---|
| [`core/`](core/README.md) | Field and gradient recovery, relative source vectors, Kabsch pose recovery, minimal synthetic example | Main localization algorithm |
| [`calibration/`](calibration/README.md) | 15 recorded CSV files, sensor mounting geometry, affine calibration, residual analysis | Main Sec. III-C, detailed calibration procedure |
| [`source_geometry/`](source_geometry/README.md) | Continuous geometry sweep and evaluation of five recorded source configurations | Main geometry study, Supplementary Sec. I, Fig. 1, Table I |
| [`distance/`](distance/README.md) | Fixed simulation geometry and source moments, synthetic distance sweep | Supplementary Sec. II and Fig. 2 |
| [`tdm_motion/`](tdm_motion/README.md) | Circular-motion sampling model, CV/CC timing, controls and plots | Supplementary Sec. III and Fig. 3 |
| [`supplementary/`](supplementary/README.md) | Clean TeX, bibliography, and figure assets | Full supplementary document including comparison Table II |
| [`docs/`](docs/reproduction_map.md) | Result mapping, provenance, and verification report | Reproduction guide |

The 15 calibration CSVs are recorded measurements. Source-geometry, distance, and motion results are numerical simulations. Fixed geometric inputs and previously fitted moment constants retain their provenance. Raw localization acquisitions and experimental localization results are not included. See [reproduction scope](docs/reproduction_scope.md).

## Start here

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 core/run_simulation.py --config core/configs/example.yaml
python3 calibration/reproduce_calibration.py --output calibration/results
```

Numerical reproduction requires NumPy and PyYAML. Python plots additionally require `requirements-plot.txt`. The supplementary document and the distance/motion paper figures use LaTeX with IEEEtran or PGFPlots as indicated in each module.

Reference outputs are preserved in each module's `reference_results/`. Write new calculations to a fresh `results/` directory or another empty directory. The numerical scripts do not contact ROS or control hardware.

See [result-to-file mapping](docs/reproduction_map.md), [provenance](docs/provenance.md), and [local verification](docs/verification.md).

## Terms of use

Copyright remains with the authors. This is a publicly accessible research repository with all rights reserved. It is not released under an open-source or open-data license. See [LICENSE](LICENSE) for the terms and permission contact.
